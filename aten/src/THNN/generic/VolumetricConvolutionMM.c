#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricConvolutionMM.c"
#else

#define CONV3D_OMP_THRESHOLD 20

static void inline THNN_(VolumetricConvolutionMM_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         THTensor *weight,
                         THTensor *bias,
                         int kT,
                         int kW,
                         int kH,
                         int dT,
                         int dW,
                         int dH,
                         int pT,
                         int pW,
                         int pH) {
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");
  THArgCheck(kT > 0 && kW > 0 && kH > 0, 8,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);

  int ndim = input->nDimension;
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (ndim == 5)
  {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  int64_t nInputPlane;
  int64_t inputDepth;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t nOutputPlane;

  int64_t exactInputDepth;
  int64_t exactInputHeight;
  int64_t exactInputWidth;
  int64_t outputDepth;
  int64_t outputHeight;
  int64_t outputWidth;

  nInputPlane = input->size[dimf];
  inputDepth = input->size[dimt];
  inputHeight  = input->size[dimh];
  inputWidth   = input->size[dimw];
  nOutputPlane = weight->size[0];

  exactInputDepth = inputDepth + 2*pT;
  exactInputHeight = inputHeight + 2*pH;
  exactInputWidth = inputWidth + 2*pW;

  if (exactInputDepth < kT || exactInputHeight < kH || exactInputWidth < kW) {
    THError("Calculated input size: (%d x %d x %d). "
      "Kernel size: (%d x %d x %d). Kernel size can't greater than actual input size",
      exactInputDepth,exactInputHeight,exactInputWidth,kT,kH,kW);
  }

  outputDepth  = (exactInputDepth - kT) / dT + 1;
  outputHeight = (exactInputHeight - kH) / dH + 1;
  outputWidth  = (exactInputWidth - kW) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1 || outputDepth < 1)
  {
    THError(
      "Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
      nInputPlane, inputDepth, inputHeight, inputWidth,
      nOutputPlane, outputDepth, outputHeight, outputWidth
    );
  }

  THArgCheck(weight->nDimension == 2 || weight->nDimension == 5, 4,
             "weight tensor should be 2D or 5D - got %d", weight->nDimension);

  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[0]);
  }

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimt, outputDepth);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

static THTensor* THNN_(view_weight)(THTensor *weight)
{
  weight = THTensor_(newContiguous)(weight);
  if (weight->nDimension == 5) {
    int64_t s1 = weight->size[0];
    int64_t s2 = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];
    THTensor *old_weight = weight;
    weight = THTensor_(newWithStorage2d)(weight->storage, weight->storageOffset,
					 s1, -1, s2, -1);
    THTensor_(free)(old_weight);
  }
  return weight;
}

/* note: due to write issues, this one cannot be parallelized as well as unfolded_copy */
static void THNN_(unfolded_acc_vol)(
          THTensor *finput,
          THTensor *input,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          long nInputPlane,
          long inputDepth,
          long inputWidth,
          long inputHeight,
          long outputDepth,
          long outputWidth,
          long outputHeight)
{
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);
#ifdef _OPENMP
  int inOmp = omp_in_parallel();
  #pragma omp parallel if (!inOmp)
  {
    size_t num_threads = omp_get_num_threads();
    size_t tid = omp_get_thread_num();
    int64_t n = nInputPlane * inputHeight * inputWidth * inputDepth;
    int64_t seg_len_tmp = n / num_threads;
    int64_t line_index_offset = tid * seg_len_tmp;
    int64_t line_seg_len = (tid == num_threads - 1)? (n-line_index_offset) : seg_len_tmp;

    long w = line_index_offset % inputWidth + pW;
    long h_index = line_index_offset / inputWidth;
    long h = h_index % inputHeight + pH;
    long d_index = h_index / inputHeight;
    long d = d_index % inputDepth + pT;
    long c = d_index / inputDepth;
#else
    int64_t line_seg_len = nInputPlane * inputHeight * inputWidth * inputDepth;
    int line_index_offset = 0;
    long w = pW;
    long h = pH;
    long d = pT;
    long c = 0;;
#endif
    long outputHW = outputHeight * outputWidth;
    long outputDHW = outputDepth * outputHW;
    long kHkW = kH*kW;
    long kTkHkW = kT*kHkW;

    register long coeff_d_col = outputHW - dT * kHkW * outputDHW;
    register long coeff_h_col = outputWidth - dH * kW * outputDHW;
    register long coeff_w_col = (1 - dW * outputDHW);

    int64_t count = 0;
    while (count < line_seg_len) {
      // compute the start and end of the output
      int w_col_start = (w < kW) ? 0 : (w - kW) / dW + 1;
      int w_col_tmp = w / dW + 1;
      int w_col_end = w_col_tmp < outputWidth? w_col_tmp : outputWidth;

      int h_col_start = (h < kH) ? 0 : (h - kH) / dH + 1;
      int h_col_tmp = h / dH + 1;
      int h_col_end = h_col_tmp < outputHeight? h_col_tmp : outputHeight;

      int d_col_start = (d < kT) ? 0 : (d - kT) / dT + 1;
      int d_col_tmp = d / dT + 1;
      int d_col_end = d_col_tmp < outputDepth? d_col_tmp : outputDepth;

      real val = 0;
      int offset = (c * kTkHkW + d * kHkW + h * kW + w) * outputDHW;

      register int offset_w_col_start = w_col_start * coeff_w_col;
      register int offset_d_col_start = d_col_start * coeff_d_col;
      register int offset_h_col_start = h_col_start * coeff_h_col;
      int offset_w_col = offset_w_col_start + offset;
      int offset_d_col;
      int offset_h_col;
      int w_col, d_col, h_col;
      for (w_col = w_col_start; w_col < w_col_end; ++w_col) {
        offset_d_col = offset_d_col_start + offset_w_col;
        for (d_col = d_col_start; d_col < d_col_end; ++d_col) {
          offset_h_col = offset_h_col_start + offset_d_col;
          for (h_col = h_col_start; h_col < h_col_end; ++h_col) {
            val += finput_data[offset_h_col];
            offset_h_col += coeff_h_col;
          }
          offset_d_col += coeff_d_col;
        }
        offset_w_col += coeff_w_col;
      }
      input_data[line_index_offset+count] = val;

      count++;
      if (count < line_seg_len) {
        if (w - pW + 1 == inputWidth) {
          w = pW;
          if (h - pH + 1 == inputHeight) {
            h = pH;
            if (d - pT + 1 == inputDepth) {
              d = pT;
              c++;
            }
            else d++;
          }
          else h++;
        }
        else w++;
      }
    }
#ifdef _OPENMP
  }
#endif
}

static void THNN_(unfolded_copy_vol)(
          THTensor *finput,
          THTensor *input,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          long nInputPlane,
          long inputDepth,
          long inputWidth,
          long inputHeight,
          long outputDepth,
          long outputWidth,
          long outputHeight)
{
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#ifdef _OPENMP
  int inOmp = omp_in_parallel();
  #pragma omp parallel if (!inOmp)
  {
    size_t num_threads = omp_get_num_threads();
    size_t tid = omp_get_thread_num();
    int64_t n = nInputPlane*kT*kH*kW;
    int64_t seg_len_tmp = n / num_threads;
    int64_t line_index_offset = tid * seg_len_tmp;
    int64_t line_seg_len = (tid == num_threads - 1)? (n-line_index_offset) : seg_len_tmp;

    long kw = line_index_offset % kW;
    long remained = line_index_offset / kW;
    long kh = remained % kH;
    remained = remained / kH;
    long kt = remained % kT;
    long nip = remained / kT;
#else
    int64_t line_seg_len = nInputPlane*kT*kH*kW;
    long kw = 0;
    long kh = 0;
    long kt = 0;
    long nip = 0;
#endif
    long dstkwStride = outputDepth*outputHeight*outputWidth;
    long dstkhStride = kW*dstkwStride;
    long dstktStride = kH*dstkhStride;
    long dstnipStride = kT*dstktStride;

    long srcnipStride = inputDepth*inputHeight*inputWidth;

    real *dst = finput_data
      + nip * dstnipStride
      + kt  * dstktStride
      + kh  * dstkhStride
      + kw  * dstkwStride;
    real *src = input_data + nip*srcnipStride;

    int64_t count = 0;
    long t,x,y,it,ix,iy;
    while (count < line_seg_len) {
      if (pT > 0 || pH > 0 || pW > 0)
      {
        for (t = 0; t < outputDepth; t++)
        {
          it = t*dT - pT + kt;
          for (y = 0; y < outputHeight; y++)
          {
            iy = y*dH - pH + kh;
            for (x = 0; x < outputWidth; x++)
            {
              ix = x*dW - pW + kw;
              if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight || ix < 0 || ix >= inputWidth)
                memset(dst+t*outputHeight*outputWidth+y*outputWidth+x, 0, sizeof(real)*(1));
              else
                memcpy(dst+t*outputHeight*outputWidth+y*outputWidth+x, src+it*inputHeight*inputWidth+iy*inputWidth+ix, sizeof(real)*(1));
            }
          }
        }
      }
      else
      {
        for (t = 0; t < outputDepth; t++)
        {
          it = t*dT + kt;
          for (y = 0; y < outputHeight; y++)
          {
            iy = y*dH + kh;
            for(x = 0; x < outputWidth; x++)
            {
              ix = x*dW + kw;
              memcpy(dst+t*outputHeight*outputWidth+y*outputWidth+x, src+it*inputHeight*inputWidth+iy*inputWidth+ix, sizeof(real)*(1));
            }
          }
        }
      }
      count++;
      if (count < line_seg_len) {
        if (kw + 1 == kW) {
          dst -= kw*dstkwStride;
          kw = 0;
          if (kh + 1 == kH) {
            dst -= kh*dstkhStride;
            kh = 0;
            if (kt  + 1== kT) {
              dst -= kt*dstktStride;
              kt = 0;
              nip++;
              dst += dstnipStride;
              src += srcnipStride;
            }
            else {
              kt++;
              dst += dstktStride;
            }
          }
          else {
            kh++;
            dst += dstkhStride;
          }
        }
        else {
          kw++;
          dst += dstkwStride;
        }
      }
    }
#ifdef _OPENMP
  }
#endif
}

static void THNN_(VolumetricConvolutionMM_updateOutput_frame)(
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int64_t nInputPlane,
          int64_t inputDepth,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t nOutputPlane,
          int64_t outputDepth,
          int64_t outputWidth,
          int64_t outputHeight)
{
  int64_t i;
  THTensor *output2d;

  THNN_(unfolded_copy_vol)(
    finput, input,
    kT, kW, kH,
    dT, dW, dH,
    pT, pW, pH,
    nInputPlane,
    inputDepth, inputWidth, inputHeight,
    outputDepth, outputWidth, outputHeight
  );

  output2d = THTensor_(newWithStorage2d)(
    output->storage, output->storageOffset, nOutputPlane, -1,
    outputDepth*outputHeight*outputWidth, -1
  );

  if (bias) {
      for (i = 0; i < nOutputPlane; i++)
      {
        THVector_(fill)(
          output->storage->data+output->storageOffset+output->stride[0]*i,
          THTensor_(get1d)(bias, i),
          outputDepth*outputHeight*outputWidth
        );
      }
  } else {
    THTensor_(zero)(output);
  }

  THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
}

void THNN_(VolumetricConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput, // unused
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  int64_t nInputPlane;
  int64_t inputDepth;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t nOutputPlane;
  int64_t outputDepth;
  int64_t outputHeight;
  int64_t outputWidth;

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, NULL, weight, bias,
        kT, kW, kH, dT, dW, dH, pT, pW, pH);
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 5)
  {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  nInputPlane = input->size[dimf];
  inputDepth = input->size[dimt];
  inputHeight  = input->size[dimh];
  inputWidth   = input->size[dimw];
  nOutputPlane = weight->size[0];
  outputDepth  = (inputDepth + 2*pT - kT) / dT + 1;
  outputHeight = (inputHeight + 2*pH - kH) / dH + 1;
  outputWidth  = (inputWidth + 2*pW - kW) / dW + 1;

  weight = THNN_(view_weight)(weight);

  if (input->nDimension == 4)
  {
    THTensor_(resize2d)(finput, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);

    THNN_(VolumetricConvolutionMM_updateOutput_frame)(
      input, output, weight, bias, finput,
      kT, kW, kH,
      dT, dW, dH,
      pT, pW, pH,
      nInputPlane, inputDepth, inputWidth, inputHeight,
      nOutputPlane, outputDepth, outputWidth, outputHeight
    );
  }
  else
  {
    int64_t T = input->size[0];
    int64_t t;

    THTensor_(resize3d)(finput, T, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize5d)(output, T, nOutputPlane, outputDepth, outputHeight, outputWidth);
#ifdef _OPENMP
    #pragma omp parallel for if(T > CONV3D_OMP_THRESHOLD) private(t)
#endif
    for (t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(VolumetricConvolutionMM_updateOutput_frame)(
        input_t, output_t, weight, bias, finput_t,
        kT, kW, kH,
        dT, dW, dH,
        pT, pW, pH,
        nInputPlane, inputDepth, inputWidth, inputHeight,
        nOutputPlane, outputDepth, outputWidth, outputHeight
      );

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }

  THTensor_(free)(input);
  THTensor_(free)(weight);
}

static void THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
          THTensor *gradInput,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *fgradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(
    gradOutput->storage, gradOutput->storageOffset,
    gradOutput->size[0], -1,
    gradOutput->size[1]*gradOutput->size[2]*gradOutput->size[3], -1
  );

  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
  THTensor_(free)(gradOutput2d);

  THTensor_(zero)(gradInput);

  THNN_(unfolded_acc_vol)(
    fgradInput, gradInput,
    kT, kW, kH,
    dT, dW, dH,
    pT, pW, pH,
    gradInput->size[0], gradInput->size[1], gradInput->size[3], gradInput->size[2],
    gradOutput->size[1], gradOutput->size[3], gradOutput->size[2]
  );
}

void THNN_(VolumetricConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  int nOutputPlane = (int)weight->size[0];

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, gradOutput, weight, NULL,
        kT, kW, kH, dT, dW, dH, pT, pW, pH);
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  weight = THNN_(view_weight)(weight);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  // depending on the BLAS library, fgradInput (result tensor) might
  // be left uninitialized on zero alpha, which might lead to weird behavior
  // hence, to be safe, zero it
  THTensor_(zero)(fgradInput);
  THTensor *tweight = THTensor_(new)();
  THTensor_(transpose)(tweight, weight, 0, 1);

  if (input->nDimension == 4)
  {
    THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
      gradInput, gradOutput, tweight, fgradInput,
      kT, kW, kH,
      dT, dW, dH,
      pT, pW, pH
    );
  }
  else
  {
    int64_t T = input->size[0];
    int64_t t;

#ifdef _OPENMP
    #pragma omp parallel for if(T > CONV3D_OMP_THRESHOLD) private(t)
#endif
    for (t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
        gradInput_t, gradOutput_t, tweight, fgradInput_t,
        kT, kW, kH,
        dT, dW, dH,
        pT, pW, pH
      );

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(free)(tweight);
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THTensor_(free)(weight);
}

static void THNN_(VolumetricConvolutionMM_accGradParameters_frame)(
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          real scale)
{
  int64_t i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(
    gradOutput->storage, gradOutput->storageOffset,
    gradOutput->size[0], -1,
    gradOutput->size[1]*gradOutput->size[2]*gradOutput->size[3], -1
  );

  THTensor *tfinput = THTensor_(new)();
  THTensor_(transpose)(tfinput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, tfinput);
  THTensor_(free)(tfinput);

  if (gradBias) {
    for (i = 0; i < gradBias->size[0]; i++)
    {
      int64_t k;
      real sum = 0;
      real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
      for (k = 0; k < gradOutput2d->size[1]; k++)
        sum += data[k];

      (gradBias->storage->data + gradBias->storageOffset)[i] += scale * sum;
    }
  }

  THTensor_(free)(gradOutput2d);
}

void THNN_(VolumetricConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  int nOutputPlane = (int)gradWeight->size[0];

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, gradOutput, gradWeight, gradBias,
        kT, kW, kH, dT, dW, dH, pT, pW, pH);
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  gradWeight = THNN_(view_weight)(gradWeight);

  if (input->nDimension == 4)   // non-batch mode
  {
    THNN_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale);
  }
  else  // batch mode
  {
    int64_t T = input->size[0];
    int64_t t;

#ifdef _OPENMP
    #pragma omp parallel for if(T > CONV3D_OMP_THRESHOLD) private(t)
#endif
    for (t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THTensor_(free)(gradWeight);
}

#endif
