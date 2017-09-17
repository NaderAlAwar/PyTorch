#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialAdaptiveMaxPooling.c"
#else

static void THNN_(SpatialAdaptiveMaxPooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          THIndex_t *ind_p,
          int64_t nslices,
          int64_t iwidth,
          int64_t iheight,
          int64_t owidth,
          int64_t oheight,
          int64_t stridew,
          int64_t strideh,
          int64_t strided)
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    int64_t i, j;
    for(i = 0; i < oheight; i++)
    {
      int y_start = (int)floor((float)i / oheight * iheight);
      int y_end   = (int)ceil((float)(i + 1) / oheight * iheight);
      int kH = y_end-y_start;

      for(j = 0; j < owidth; j++)
      {

        int x_start = (int)floor((float)j / owidth * iwidth);
        int x_end   = (int)ceil((float)(j + 1) / owidth * iwidth);
        int kW = x_end-x_start;

        /* local pointers */
        real *ip = input_p   + k*strided + y_start*strideh + x_start*stridew;
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        THIndex_t *indp = ind_p   + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        int64_t maxindex = -1;
        real maxval = -FLT_MAX;
        int64_t tcntr = 0;
        int x,y;
        for(y = 0; y < kH; y++)
        {
          for(x = 0; x < kW; x++)
          {
            real val = *(ip + y*strideh + x*stridew);
            if (val > maxval)
            {
              maxval = val;
              maxindex = (y+y_start)*iwidth + (x+x_start);
            }
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max */
        *indp = maxindex + TH_INDEX_BASE;
      }
    }
  }
}

void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int owidth,
          int oheight)
{
  int dimw = 2;
  int dimh = 1;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iheight;
  int64_t iwidth;

  int64_t istride_d;
  int64_t istride_h;
  int64_t istride_w;
  int64_t istride_b;

  real *input_data;
  real *output_data;
  THIndex_t *indices_data;


  THNN_ARGCHECK(input->nDimension == 3 || input->nDimension == 4, 2, input,
		"3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->nDimension == 4)
  {
    istride_b = input->stride[0];
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  /* strides */
  istride_d = input->stride[dimh-1];
  istride_h = input->stride[dimh];
  istride_w = input->stride[dimw];

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);
    /* indices will contain i,j locations for each output point */
    THIndexTensor_(resize3d)(indices, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    THNN_(SpatialAdaptiveMaxPooling_updateOutput_frame)(input_data, output_data,
                                                      indices_data,
                                                      nslices,
                                                      iwidth, iheight,
                                                      owidth, oheight,
                                                      istride_w,istride_h,
                                                      istride_d);
  }
  else
  {
    int64_t p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
    /* indices will contain i,j locations for each output point */
    THIndexTensor_(resize4d)(indices, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialAdaptiveMaxPooling_updateOutput_frame)(input_data+p*istride_b, output_data+p*nslices*owidth*oheight,
                                                        indices_data+p*nslices*owidth*oheight,
                                                        nslices,
                                                        iwidth, iheight,
                                                        owidth, oheight,
                                                        istride_w,istride_h,
                                                        istride_d);
    }
  }
}

static void THNN_(SpatialAdaptiveMaxPooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          THIndex_t *ind_p,
          int64_t nslices,
          int64_t iwidth,
          int64_t iheight,
          int64_t owidth,
          int64_t oheight)
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k*iwidth*iheight;
    real *gradOutput_p_k = gradOutput_p + k*owidth*oheight;
    THIndex_t *ind_p_k = ind_p + k*owidth*oheight;

    /* calculate max points */
    int64_t i, j;
    for(i = 0; i < oheight; i++)
    {
      int y_start = (int)floor((float) i / oheight * iheight);
      for(j = 0; j < owidth; j++)
      {
        int x_start = (int)floor((float) j / owidth * iwidth);
        /* retrieve position of max */
        int64_t maxp = ind_p_k[i*owidth + j] - TH_INDEX_BASE;

        /* update gradient */
        gradInput_p_k[maxp] += gradOutput_p_k[i*owidth + j];
      }
    }
  }
}

void THNN_(SpatialAdaptiveMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices)
{
  int dimw = 2;
  int dimh = 1;
  int64_t nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;
  THIndex_t *indices_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THIndexTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 3)
  {
    THNN_(SpatialAdaptiveMaxPooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                           indices_data,
                                                           nslices,
                                                           iwidth, iheight,
                                                           owidth, oheight);
  }
  else
  {
    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialAdaptiveMaxPooling_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, gradOutput_data+p*nslices*owidth*oheight,
                                                             indices_data+p*nslices*owidth*oheight,
                                                             nslices,
                                                             iwidth, iheight,
                                                             owidth, oheight);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
