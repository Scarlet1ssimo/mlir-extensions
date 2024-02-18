//===- VectorToXeGPU.cpp - VectorToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the VectorToXeGPU conversion, converting the Vector
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/VectorToXeGPU/VectorToXeGPU.h>
#include <imex/Dialect/XeGPU/IR/XeGPU.h>
#include <imex/Utils/PassWrapper.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include <mlir/IR/BuiltinOps.h>

#include "../PassDetail.h"
#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/VectorInterfaces.h"

using namespace mlir;

namespace imex {

namespace {

class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

struct MyTarget : public ConversionTarget {
  MyTarget(MLIRContext &ctx) : ConversionTarget(ctx) {

    /// Mark `cf.br` and `cf.cond_br` as illegal.
    addIllegalOp<vector::TransferReadOp>(); //, vector::TransferWriteOp
  }
};

// *******************************
// ***** Individual patterns *****
// *******************************

// Goal: vector.transfer_read -> xegpu.create_nd_tdesc + xegpu.load_nd
// E.g.
// %3 = vector.transfer_read %arg1[%0, %2], %arg2 : memref<512x640xf32>,
// vector<32xf32> to %desc = xegpu.create_nd_tdesc %arg1[%0, %2] {mode = vc} :
// memref<512x640xf32> -> !xegpu.tensor_desc<32xf32> %3 = xegpu.load_nd %desc
// {mode = vc}: !xegpu.tensor_desc<32xf32> -> vector<32xf32>

struct TransferReadOpConverter
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    auto resultTile = read.getResult();
    auto resTileType = resultTile.getType();
    auto resTileShape = resTileType.getShape();
    auto source = read.getSource();
    llvm::errs() << "----------------------\n";
    llvm::errs() << __LINE__ << ": " << resultTile << "\n"; // %3 = ...
    llvm::errs() << __LINE__ << ": " << resTileType << "\n"; // vector<32xf32>
    llvm::errs() << __LINE__ << ": " << source << "\n";//memref<512x640xf32>
    for (auto op : read->getOperands()) {
      llvm::errs() << __LINE__ << ": " << op << "\n";
      /*
      <block argument> of type 'memref<512x640xi32>' at index: 1
      %0 = gpu.block_id  x
      %2 = arith.muli %1, %arg0 : index
      <block argument> of type 'i32' at index: 2
      */
    }

    auto tDescTy = xegpu::TensorDescType::get({resTileShape[0]},
                                              resTileType.getElementType());
    mlir::SmallVector<mlir::OpFoldResult> tDescOffsets{read->getOperand(1),
                                                       read->getOperand(2)};

    rewriter.setInsertionPoint(read);
    auto desc = rewriter.create<xegpu::CreateNdDescOp>(
        read.getLoc(), tDescTy, source, tDescOffsets, imex::xegpu::Mode::VC);
    mlir::IntegerAttr vnniAxisAttr;
    mlir::DenseI64ArrayAttr transposeAttr;
    auto L1 = xegpu::CacheReadHintAttr::get(read.getContext(),
                                            xegpu::CacheReadHint::CACHED);
    auto L2 = xegpu::CacheReadHintAttr::get(read.getContext(),
                                            xegpu::CacheReadHint::CACHED);
    auto L3 = xegpu::CacheReadHintAttr::get(read.getContext(),
                                            xegpu::CacheReadHint::CACHED);
    auto load = rewriter.create<xegpu::LoadNDOp>(
        read.getLoc(), resTileType, desc, vnniAxisAttr, transposeAttr, L1, L2,
        L3, imex::xegpu::Mode::VC);
    llvm::errs() << "end----------------------\n";
    rewriter.replaceOp(read, load->getResults());
    return ::mlir::success();
  }
};

// vector.transfer_write %5, %arg4[%0, %2] : vector<32xf32>, memref<512x640xf32>
// to
// %desc2 = xegpu.create_nd_tdesc %arg4[%0, %2] {mode = vc} : memref<512x640xf32> -> !xegpu.tensor_desc<32xf32>
// xegpu.store_nd %5, %desc2 {mode = vc} : vector<32xf32>, !xegpu.tensor_desc<32xf32>
struct TransferWriteOpConverter
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "----------------------\n";
    auto resultTile = write->getOperand(0);//%5
    auto source = write.getSource(); // memref<512x640xi32>
    llvm::errs() << __LINE__ << ": " << source << "\n";

    auto resTileType = dyn_cast<VectorType>(resultTile.getType());
    llvm::errs() << __LINE__ << ": " << resTileType << "\n";
    auto resTileShape = resTileType.getShape();
    for (auto op : write->getOperands()) {
      llvm::errs() << __LINE__ << ": " << op << "\n";
    }

    auto tDescTy = xegpu::TensorDescType::get({resTileShape[0]},
                                              resTileType.getElementType());
    mlir::SmallVector<mlir::OpFoldResult> tDescOffsets{write->getOperand(2),
                                                       write->getOperand(3)};

    rewriter.setInsertionPoint(write);
    auto desc = rewriter.create<xegpu::CreateNdDescOp>(write.getLoc(), tDescTy,
                                                       source, tDescOffsets,
                                                       imex::xegpu::Mode::VC);
    llvm::errs() << __LINE__ << ": " << desc << "\n";
    auto L1 = xegpu::CacheWriteHintAttr::get(write->getContext(),
                                             xegpu::CacheWriteHint::WRITE_BACK);
    auto L2 = xegpu::CacheWriteHintAttr::get(write->getContext(),
                                             xegpu::CacheWriteHint::WRITE_BACK);
    auto L3 = xegpu::CacheWriteHintAttr::get(write->getContext(),
                                             xegpu::CacheWriteHint::WRITE_BACK);
    rewriter.create<xegpu::StoreNDOp>(write.getLoc(), TypeRange(), desc,
                                      write->getOperand(0), L1, L2, L3,
                                      imex::xegpu::Mode::VC);
    llvm::errs() << "end----------------------\n";
    rewriter.eraseOp(write);
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct ConvertVectorToXeGPUPass // convert Vector to XeGPU
    : public ::imex::ConvertVectorToXeGPUBase<ConvertVectorToXeGPUPass> {
  ConvertVectorToXeGPUPass() = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<TransferReadOpConverter, TransferWriteOpConverter>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
    llvm::errs() << "End!!!\n";
    llvm::errs() << getOperation() << "\n";
  }
};

} // namespace

/// Populate the given list with patterns that convert Vector to XeGPU

/// Create a pass that convert Vector to XeGPU
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertVectorToXeGPUPass() {
  return std::make_unique<ConvertVectorToXeGPUPass>();
}

} // namespace imex
