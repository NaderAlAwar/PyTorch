#include <iostream>

#include "caffe2/core/operator.h"
#include <gtest/gtest.h>


namespace caffe2 {

class WorkspaceTestFoo {};

CAFFE_KNOWN_TYPE(WorkspaceTestFoo);

TEST(WorkspaceTest, BlobAccess) {
  Workspace ws;

  EXPECT_FALSE(ws.HasBlob("nonexisting"));
  EXPECT_EQ(ws.GetBlob("nonexisting"), nullptr);

  EXPECT_EQ(ws.GetBlob("newblob"), nullptr);
  EXPECT_NE(nullptr, ws.CreateBlob("newblob"));
  EXPECT_NE(nullptr, ws.GetBlob("newblob"));
  EXPECT_TRUE(ws.HasBlob("newblob"));

  // Different names should still be not created.
  EXPECT_FALSE(ws.HasBlob("nonexisting"));
  EXPECT_EQ(ws.GetBlob("nonexisting"), nullptr);

  // Check if the returned Blob is OK for all operations
  Blob* blob = ws.GetBlob("newblob");
  int* int_unused UNUSED_VARIABLE = blob->GetMutable<int>();
  EXPECT_TRUE(blob->IsType<int>());
  EXPECT_FALSE(blob->IsType<WorkspaceTestFoo>());
  EXPECT_NE(&blob->Get<int>(), nullptr);

  // Re-creating the blob does not change the content as long as it already
  // exists.
  EXPECT_NE(nullptr, ws.CreateBlob("newblob"));
  EXPECT_TRUE(blob->IsType<int>());
  EXPECT_FALSE(blob->IsType<WorkspaceTestFoo>());
  // When not null, we should only call with the right type.
  EXPECT_NE(&blob->Get<int>(), nullptr);

  // test removing blob
  EXPECT_FALSE(ws.HasBlob("nonexisting"));
  EXPECT_FALSE(ws.RemoveBlob("nonexisting"));
  EXPECT_TRUE(ws.HasBlob("newblob"));
  EXPECT_TRUE(ws.RemoveBlob("newblob"));
  EXPECT_FALSE(ws.HasBlob("newblob"));
}

TEST(WorkspaceTest, RunEmptyPlan) {
  PlanDef plan_def;
  Workspace ws;
  EXPECT_TRUE(ws.RunPlan(plan_def));
}

TEST(WorkspaceTest, Sharing) {
  Workspace parent;
  EXPECT_FALSE(parent.HasBlob("a"));
  EXPECT_TRUE(parent.CreateBlob("a"));
  EXPECT_TRUE(parent.GetBlob("a"));
  {
    Workspace child(&parent);
    // Child can access parent blobs
    EXPECT_TRUE(child.HasBlob("a"));
    EXPECT_TRUE(child.GetBlob("a"));
    // Child can create local blobs
    EXPECT_FALSE(child.HasBlob("b"));
    EXPECT_FALSE(child.GetBlob("b"));
    EXPECT_TRUE(child.CreateBlob("b"));
    EXPECT_TRUE(child.GetBlob("b"));
    // Parent cannot access child blobs
    EXPECT_FALSE(parent.GetBlob("b"));
    EXPECT_FALSE(parent.HasBlob("b"));
    // Parent can create duplicate names
    EXPECT_TRUE(parent.CreateBlob("b"));
    // But child has local overrides
    EXPECT_NE(child.GetBlob("b"), parent.GetBlob("b"));
  }
}

}  // namespace caffe2
