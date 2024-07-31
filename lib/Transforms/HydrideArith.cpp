//===- OpStats.cpp - Prints stats of operations in module -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "imex/Transforms/HydrideUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <queue>
#include <string>

using llvm::dbgs;

#define DEBUG_TYPE "hydride"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << ":" << __LINE__ << "] ")

using namespace mlir;
using namespace imex;

unsigned op_lanes = 0;

namespace {

class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

class HydrideSynthEmitter {
public:
  HydrideSynthEmitter(
      std::string benchmark_name,
      std::map<vector::LoadOp, unsigned> &LoadToRegMap,
      std::map<void *, unsigned> &VariableToRegMap,
      std::map<vector::TransferReadOp, unsigned> &TransferReadToRegMap)
      : benchmark_name(benchmark_name), LoadToRegMap(LoadToRegMap),
        VariableToRegMap(VariableToRegMap),
        TransferReadToRegMap(TransferReadToRegMap) {}

  std::string benchmark_name;
  std::map<vector::LoadOp, unsigned> LoadToRegMap;
  std::map<void *, unsigned> VariableToRegMap;
  std::map<vector::TransferReadOp, unsigned> TransferReadToRegMap;

  std::string emit_set_current_bitwidth() {
    const char *bitwidth = getenv("HL_SYNTH_BW");
    if (bitwidth) {
      std::string str = bitwidth;
      int bw = stoi(str);

      if (bw > 0) {
        return "(current-bitwidth " + std::to_string(bw) + ")";
      }
    }
    return "";
  }

  std::string emit_set_memory_limit(size_t MB) {
    return "(custodian-limit-memory (current-custodian) (* " +
           std::to_string(MB) + " 1024 1024))";
  }

  std::string mlir_type_to_synth_elem(Type type, bool include_space,
                                      bool c_plus_plus) {
    bool needs_space = true;
    std::ostringstream oss;
    if (auto intType = dyn_cast<IntegerType>(type)) {
      if (intType.isUnsigned()) {
        oss << "u";
      }
      oss << "int" << std::to_string(intType.getWidth());
    }

    if (include_space && needs_space) {
      oss << " ";
    }
    return oss.str();
  }

  std::string get_synthlog_hash_filepath(int id) {
    if (id < 0) {
      const char *initial_hash = getenv("HYDRIDE_INITIAL_HASH");
      if (initial_hash) {
        return std::string(initial_hash) + ".rkt";

      } else {
        return "";
      }

    } else {
      return "hydride_hash_" + std::to_string(id) + ".rkt";
    }
  }

  std::string get_synthlog_hash_name(int id) {

    if (id < 0) {
      const char *initial_hash = getenv("HYDRIDE_INITIAL_HASH");
      if (initial_hash) {
        return std::string(initial_hash);
      } else {
        return "";
      }
    } else {
      return "synth_hash_" + std::to_string(id);
    }
  }

  std::string define_buffer(const std::string &reg_name,
                            const std::string &id_name, size_t bitwidth,
                            const std::string &elemT, unsigned rank,
                            const std::vector<unsigned> &shape) {
    std::string define_bitvector_str = "(define " + reg_name + "_tensor" + " " +
                                       "(bv 0 (bitvector " +
                                       std::to_string(bitwidth) + ")" + "))";

    std::string define_buffer_str = "(define " + reg_name +
                                    " (arith:create-tensor " + reg_name +
                                    "_tensor " + "(vector";

    for (unsigned d = 0; d < rank; d++) {
      define_buffer_str += " " + std::to_string(shape[d]);
    }

    define_buffer_str += ") ";

    define_buffer_str += "(vector";

    for (unsigned d = 0; d < rank; d++) {
      define_buffer_str += " " + std::to_string(d);
    }

    define_buffer_str += ") ";

    define_buffer_str += elemT + " ";
    define_buffer_str += id_name + ")" + ")";

    return define_bitvector_str + "\n" + define_buffer_str;
  }

  std::string define_buffer_common(const std::string &reg_name,
                                   const std::string &id_name, Type type) {
    std::string elemT = "'";
    size_t bitwidth = 0;
    unsigned rank = 0;
    std::vector<unsigned> shape;
    DBGS() << type << "\n";
    auto vecType = dyn_cast<VectorType>(type);
    if (vecType && vecType.getElementType()) {
      bitwidth = vecType.getElementTypeBitWidth() * vecType.getNumElements();
      elemT += mlir_type_to_synth_elem(vecType.getElementType(), false, true);
      rank = vecType.getRank();
      shape = std::vector<unsigned>(vecType.getShape().begin(),
                                    vecType.getShape().end());
    }

    return define_buffer(reg_name, id_name, bitwidth, elemT, rank, shape);
  }

  std::string define_load_buffer(vector::LoadOp op) {
    std::string reg_name = "reg_" + std::to_string(LoadToRegMap[op]);
    std::string id_name = std::to_string(LoadToRegMap[op]);
    return define_buffer_common(reg_name, id_name, op.getResult().getType());
  }

  std::string define_transfer_buffer(vector::TransferReadOp op) {
    std::string reg_name = "reg_" + std::to_string(TransferReadToRegMap[op]);
    std::string id_name = std::to_string(TransferReadToRegMap[op]);
    return define_buffer_common(reg_name, id_name, op.getVector().getType());
  }

  std::string define_variable_buffer(void *val_ptr) {
    auto val = Value::getFromOpaquePointer(val_ptr);
    std::string reg_name = "reg_" + std::to_string(VariableToRegMap[val_ptr]);
    std::string id_name = std::to_string(VariableToRegMap[val_ptr]);
    DBGS() << "Emit buffer for " << val << "\n";
    return define_buffer_common(reg_name, id_name, val.getType());
  }

  std::string get_reg_id(int reg_id) {
    return "(bv " + std::to_string(reg_id) + " (bitvector 8))";
  }

  std::string emit_buffer_id_map(std::string map_name) {
    std::string comment =
        "; Creating a map between buffers and mlir call node arguments\n";
    std::string define_buff_map =
        "(define " + map_name + " (make-hash))" + "\n";
    std::string add_entry = "";

    for (auto bi = LoadToRegMap.begin(); bi != LoadToRegMap.end(); bi++) {
      unsigned id = bi->second;
      add_entry += "(hash-set! " + map_name + " " + "reg_" +
                   std::to_string(id) + " " + get_reg_id(id) + ")" + "\n";
    }

    for (auto bi = TransferReadToRegMap.begin();
         bi != TransferReadToRegMap.end(); bi++) {
      unsigned id = bi->second;
      add_entry += "(hash-set! " + map_name + " " + "reg_" +
                   std::to_string(id) + " " + get_reg_id(id) + ")" + "\n";
    }

    for (auto bi = VariableToRegMap.begin(); bi != VariableToRegMap.end();
         bi++) {
      unsigned id = bi->second;
      add_entry += "(hash-set! " + map_name + " " + "reg_" +
                   std::to_string(id) + " " + get_reg_id(id) + ")" + "\n";
    }

    return comment + define_buff_map + add_entry;
  }

  std::string emit_symbolic_buffers() {
    std::string buffers = "";
    for (auto bi = LoadToRegMap.begin(); bi != LoadToRegMap.end(); bi++) {
      auto loadOp = bi->first;
      buffers += define_load_buffer(loadOp) + "\n";
    }

    for (auto bi = TransferReadToRegMap.begin();
         bi != TransferReadToRegMap.end(); bi++) {
      auto transferReadOp = bi->first;
      buffers += define_transfer_buffer(transferReadOp) + "\n";
    }

    for (auto bi = VariableToRegMap.begin(); bi != VariableToRegMap.end();
         bi++) {
      auto val = bi->first;
      buffers += define_variable_buffer(val) + "\n";
    }

    return buffers;
  }

  std::string emit_symbolic_buffers_vector(std::string vector_name) {
    std::string buffers = "(define " + vector_name + " (vector ";
    for (auto bi = LoadToRegMap.begin(); bi != LoadToRegMap.end(); bi++) {
      auto loadOp = bi->first;
      buffers += "reg_" + std::to_string(LoadToRegMap[loadOp]) + " ";
    }

    for (auto bi = TransferReadToRegMap.begin();
         bi != TransferReadToRegMap.end(); bi++) {
      auto transferReadOp = bi->first;
      buffers +=
          "reg_" + std::to_string(TransferReadToRegMap[transferReadOp]) + " ";
    }

    for (auto bi = VariableToRegMap.begin(); bi != VariableToRegMap.end();
         bi++) {
      auto val = bi->first;
      buffers += "reg_" + std::to_string(VariableToRegMap[val]) + " ";
    }

    buffers += "))";
    return buffers;
  }

  std::string emit_racket_imports() {
    return "#lang rosette\n \
(require rosette/lib/synthax)\n \
(require rosette/lib/angelic)\n \
(require racket/pretty)\n \
(require data/bit-vector)\n \
(require rosette/lib/destruct)\n \
(require rosette/solver/smt/boolector)\n \
(require hydride)\n ";
  }

  std::string emit_hydride_synthesis(std::string expr_name, size_t expr_depth,
                                     size_t VF, std::string id_map_name,
                                     std::string synth_log_path,
                                     std::string synth_log_name,
                                     std::string synth_target) {

    std::string solver = "'z3";
    const char *hydride_solver = getenv("HYDRIDE_SOLVER");

    if (hydride_solver) {
      solver = hydride_solver;
    }

    std::string optimize = "#t";
    std::string symbolic = "#f";

    return "(synthesize-mlir-expr " + expr_name + " " + id_map_name + " " +
           std::to_string(expr_depth) + " " + std::to_string(VF) + " " +
           solver + " " + optimize + " " + symbolic + "  \"" + synth_log_path +
           "\"  \"" + synth_log_name + "\"  \"" + synth_target + "\")";
  }

  std::string emit_interpret_expr(std::string expr_name) {
    return "(arith:interpret " + expr_name + ")";
  }

  std::string emit_assemble_result(std::string result_name,
                                   std::string expr_name, size_t lanes) {
    return "(define " + result_name + " (arith:assemble-bitvector " +
           emit_interpret_expr(expr_name) + " " + std::to_string(lanes) + ")" +
           ")";
  }

  std::string emit_write_synth_log_to_file(std::string fpath,
                                           std::string hash_name) {
    return "(save-synth-map \"" + fpath + "\" \"" + hash_name + "\" synth-log)";
  }

  std::string emit_compile_to_llvm(std::string expr_name, std::string map_name,
                                   std::string call_name,
                                   std::string bitcode_path) {
    std::string comment =
        "; Translate synthesized hydride-expression into LLVM-IR";
    std::string command = "(compile-to-llvm " + expr_name + " " + map_name +
                          " \"" + call_name + "\" \"" + bitcode_path + "\")\n";

    return comment + "\n" + command;
  }
};

// util functions

struct HydrideArithPass : public HydrideArithBase<HydrideArithPass> {

  std::map<unsigned, void *> RegToVariableMap;
  std::map<void *, unsigned> VariableToRegMap;

  std::map<unsigned, vector::LoadOp> RegToLoadMap;
  std::map<vector::LoadOp, unsigned> LoadToRegMap;

  std::map<unsigned, vector::TransferReadOp> RegToTransferReadMap;
  std::map<vector::TransferReadOp, unsigned> TransferReadToRegMap;

  std::queue<Operation *> RootExprOp;

  bool DisableSynth = false;
  bool DisableLegalize = false;

  const char *benchmark_name;

  HydrideArithPass() {
    if (getenv("HYDRIDE_DISABLE_SYNTH")) {
      DBGS() << "Disabling Synthesis... This will lead to linking with old "
                "files.\n";
      DisableSynth = true;
    }
    if (getenv("HYDRIDE_DISABLE_LEGALIZE")) {
      DBGS() << "Disabling Synthesis\n";
      DisableLegalize = true;
    }
    benchmark_name = getenv("HYDRIDE_BENCHMARK");
  }

  explicit HydrideArithPass(std::string synth_target) : HydrideArithPass() {
    this->synth_target = synth_target;
  }

  // Prints the resultant operation statistics post iterating over the module.
  void runOnOperation() override;

  // Print summary of op stats.
  void EmitSynthesis(std::string benchmark_name, std::string expr,
                     unsigned expr_id);

  void EmitLLVMCode();

  FlatSymbolRefAttr getOrInsertHydrideFunc(MyPatternRewriter &rewriter,
                                           gpu::GPUModuleOp module, int expr_id,
                                           std::string curr_func_name,
                                           Type retType,
                                           std::vector<Type> argTypes);

  std::string MLIRValVisit(Value val);

  std::string MLIRArithOpVisit(Operation *op);

  std::string MLIRVectorOpVisit(Operation *op);

  std::string print_binary_op(std::string bv_name, Value a, Value b,
                              bool is_vector_op);

protected:
  std::string visit(arith::ConstantOp constantOp) {
    auto constInt = cast<IntegerAttr>(constantOp.getValue()).getInt();
    return std::to_string(constInt);
  }

  std::string visit(arith::AddIOp addOp) {
    Value a = addOp.getLhs();
    Value b = addOp.getRhs();
    bool is_vec = isa<VectorType>(addOp.getResult().getType());
    std::string ret_str = print_binary_op("add", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::DivUIOp divUIOp) {
    Value a = divUIOp.getLhs();
    Value b = divUIOp.getRhs();
    bool is_vec = isa<VectorType>(divUIOp.getResult().getType());
    std::string ret_str = print_binary_op("div", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::DivSIOp divSIOp) {
    Value a = divSIOp.getLhs();
    Value b = divSIOp.getRhs();
    bool is_vec = isa<VectorType>(divSIOp.getResult().getType());
    std::string ret_str = print_binary_op("div", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::SubIOp subOp) {
    Value a = subOp.getLhs();
    Value b = subOp.getRhs();
    bool is_vec = isa<VectorType>(subOp.getResult().getType());
    std::string ret_str = print_binary_op("sub", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::MinUIOp minUIOp) {
    Value a = minUIOp.getLhs();
    Value b = minUIOp.getRhs();
    bool is_vec = isa<VectorType>(minUIOp.getResult().getType());
    std::string ret_str = print_binary_op("min", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::MinSIOp minSIOp) {
    Value a = minSIOp.getLhs();
    Value b = minSIOp.getRhs();
    bool is_vec = isa<VectorType>(minSIOp.getResult().getType());
    std::string ret_str = print_binary_op("min", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::MaxUIOp maxUIOp) {
    Value a = maxUIOp.getLhs();
    Value b = maxUIOp.getRhs();
    bool is_vec = isa<VectorType>(maxUIOp.getResult().getType());
    std::string ret_str = print_binary_op("max", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::MaxSIOp maxSIOp) {
    Value a = maxSIOp.getLhs();
    Value b = maxSIOp.getRhs();
    bool is_vec = isa<VectorType>(maxSIOp.getResult().getType());
    std::string ret_str = print_binary_op("max", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::MulIOp mulOp) {
    Value a = mulOp.getLhs();
    Value b = mulOp.getRhs();
    bool is_vec = isa<VectorType>(mulOp.getResult().getType());
    std::string ret_str = print_binary_op("mul", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::ShRSIOp shrSIOp) {
    Value a = shrSIOp.getLhs();
    Value b = shrSIOp.getRhs();
    bool is_vec = isa<VectorType>(shrSIOp.getResult().getType());
    std::string ret_str = print_binary_op("shr", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::ShRUIOp shrUIOp) {
    Value a = shrUIOp.getLhs();
    Value b = shrUIOp.getRhs();
    bool is_vec = isa<VectorType>(shrUIOp.getResult().getType());
    std::string ret_str = print_binary_op("shr", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::ShLIOp shlIOp) {
    Value a = shlIOp.getLhs();
    Value b = shlIOp.getRhs();
    bool is_vec = isa<VectorType>(shlIOp.getResult().getType());
    std::string ret_str = print_binary_op("shl", a, b, is_vec);
    return ret_str;
  }

  std::string visit(arith::ExtSIOp extSIop) {
    Value a = extSIop.getIn();
    std::string source = MLIRValVisit(a);
    if (auto vecType = dyn_cast<VectorType>(extSIop.getResult().getType())) {
      auto Ty = vecType.getElementType().getIntOrFloatBitWidth();
      auto lanes = vecType.getNumElements();
      std::string ret_str = "(arith:cast-int " + source  +
                            std::to_string(lanes) + " " + std::to_string(Ty) +
                            ")";
      return ret_str;
    }
    DBGS() << "Unsupported ExtSIOp\n";
    return "";
  }

  // TODO:ExtUIOp

  std::string visit(arith::CmpIOp cmpOp) {
    Value a = cmpOp.getOperand(0);
    Value b = cmpOp.getOperand(1);
    bool is_vec = dyn_cast<VectorType>(a.getType()) ? true : false;
    auto pred = cmpOp.getPredicate();
    std::string ret_str;
    // signed.unsigned
    switch (pred) {
    case (arith::CmpIPredicate::eq): {
      ret_str = print_binary_op("eq", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::ne): {
      ret_str = print_binary_op("ne", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::uge): {
      ret_str = print_binary_op("ge", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::ugt): {
      ret_str = print_binary_op("gt", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::ule): {
      ret_str = print_binary_op("le", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::ult): {
      ret_str = print_binary_op("lt", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::sge): {
      ret_str = print_binary_op("ge", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::sgt): {
      ret_str = print_binary_op("gt", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::sle): {
      ret_str = print_binary_op("le", a, b, is_vec);
      return ret_str;
    }
    case (arith::CmpIPredicate::slt): {
      ret_str = print_binary_op("lt", a, b, is_vec);
      return ret_str;
    }
    }
    DBGS() << "Unsupported Predicate"
           << "\n";
    return "";
  }

  std::string visit(vector::BitCastOp bitcastOp) {
    // get type of result and source
    auto castSrcEltType = bitcastOp.getSourceVectorType().getElementType();
    auto castDstEltType = bitcastOp.getResultVectorType().getElementType();
    std::string source = MLIRValVisit(bitcastOp.getSource());
    // (arith:tensor-bitcast source-elt-ty dest-elt-ty source-tensor)
    std::string ret_str =
        "(vector:bitcast (" + mlir::stringifyType(castSrcEltType) + " " +
        mlir::stringifyType(castDstEltType) + +" " + source + ") )";
    return ret_str;
  }

  std::string visit(vector::BroadcastOp broadcastOp) {
    // arith:tensor-broadcast (n vec)
    // are these semantics correct? should be the vector type
    VectorType broadcastType = broadcastOp.getResultVectorType();
    unsigned rank = broadcastType.getRank();
    std::string source = MLIRValVisit(broadcastOp.getSource());
    std::string ret_str =
        "(vector:broadcast (" + std::to_string(rank) + " " + source + ") )";
    return ret_str;
  }

  std::string visit(vector::ExtractElementOp extractElementOp) {
    auto extractPos = extractElementOp.getPosition();

    Value src = extractElementOp.getVector();
    std::string val_str;
    llvm::raw_string_ostream val_ostream(val_str);
    val_ostream << extractPos;
    std::string ret_str = "(vector:extract-element (" + MLIRValVisit(src) +
                          " " + val_ostream.str() + ") )";
    return ret_str;
  }

  std::string visit(vector::ExtractOp extractOp) {

    auto extractPos = extractOp.getStaticPosition();

    std::string RosettePosVec = "(vector";

    for (auto d : extractPos) {

      RosettePosVec += " " + std::to_string(d);
    }

    RosettePosVec += ")";

    Value src = extractOp.getVector();

    std::string ret_str =
        "(vector:extract " + MLIRValVisit(src) + " " + RosettePosVec + ")";
    return ret_str;
  }

  std::string visit(vector::ExtractStridedSliceOp extractStridedSliceOp) {

    auto sliceOffsets =
        extractVector<int64_t>(extractStridedSliceOp.getOffsets());
    auto sliceSizes = extractVector<int64_t>(extractStridedSliceOp.getSizes());
    auto sliceStrides =
        extractVector<int64_t>(extractStridedSliceOp.getStrides());

    std::string RosetteOffsetVec = "(vector";
    std::string RosetteSliceVec = "(vector";
    std::string RosetteStrideVec = "(vector";

    for (auto d : sliceOffsets) {

      RosetteOffsetVec += " " + std::to_string(d);
    }
    for (auto d : sliceSizes) {

      RosetteSliceVec += " " + std::to_string(d);
    }
    for (auto d : sliceStrides) {

      RosetteStrideVec += " " + std::to_string(d);
    }

    RosetteOffsetVec += ")";
    RosetteSliceVec += ")";
    RosetteStrideVec += ")";

    Value src = extractStridedSliceOp.getVector();

    std::string ret_str = "(vector:extract_strided_slice " + MLIRValVisit(src) +
                          " " + RosetteOffsetVec + " " + RosetteSliceVec + " " +
                          RosetteStrideVec + " " + ")";
    return ret_str;
  }

  std::string visit(vector::InsertStridedSliceOp insertStridedSliceOp) {

    auto sliceOffsets =
        extractVector<int64_t>(insertStridedSliceOp.getOffsets());
    auto sliceStrides =
        extractVector<int64_t>(insertStridedSliceOp.getStrides());

    std::string RosetteOffsetVec = "(vector";
    std::string RosetteStrideVec = "(vector";

    for (auto d : sliceOffsets) {

      RosetteOffsetVec += " " + std::to_string(d);
    }

    for (auto d : sliceStrides) {

      RosetteStrideVec += " " + std::to_string(d);
    }

    RosetteOffsetVec += ")";
    RosetteStrideVec += ")";

    Value src = insertStridedSliceOp.getSource();
    Value dst = insertStridedSliceOp.getDest();

    std::string ret_str = "(vector:insert-strided-slice " + MLIRValVisit(src) +
                          " " + MLIRValVisit(dst) + " " + RosetteOffsetVec +
                          " " + RosetteStrideVec + " " + ")";
    return ret_str;
  }

  std::string visit(vector::FMAOp fmaOp) {

    Value lhs = fmaOp.getLhs();
    Value rhs = fmaOp.getRhs();
    Value acc = fmaOp.getAcc();
    std::string mul_str = print_binary_op("mul", lhs, rhs, true);
    std::string ret_str =
        "(arith:tensor-add " + mul_str + " " + MLIRValVisit(acc) + ")\n";
    return ret_str;
  }

  std::string visit(vector::FlatTransposeOp flatTransposeOp) {
    unsigned rows = flatTransposeOp.getRows();
    unsigned cols = flatTransposeOp.getColumns();
    Value src = flatTransposeOp.getMatrix();

    std::string ret_str = "(vector:flat-transpose " + std::to_string(rows) +
                          " " + std::to_string(cols) + " " + MLIRValVisit(src) +
                          " " + ")";
    return ret_str;
  }
  std::string visit(vector::MatmulOp matMulOp) {
    unsigned lhs_rows = matMulOp.getLhsRows();
    unsigned lhs_columns = matMulOp.getLhsColumns();
    unsigned rhs_columns = matMulOp.getRhsColumns();
    Value lhs = matMulOp.getLhs();
    Value rhs = matMulOp.getRhs();
    std::string ret_str = "(vector:matmul " + std::to_string(lhs_rows) + " " +
                          std::to_string(lhs_columns) + " " +
                          std::to_string(rhs_columns) + " " +
                          MLIRValVisit(lhs) + " " + MLIRValVisit(rhs) + ")";
    return ret_str;
  }

  std::string visit(vector::ReductionOp reductionOp) {
    auto src = reductionOp.getVector();
    auto kind = reductionOp.getKind();
    std::string kind_str;
    switch (kind) {
    case vector::CombiningKind::ADD:
      kind_str = "'add";
      break;
    case vector::CombiningKind::MUL:
      kind_str = "'mul";
      break;
    case vector::CombiningKind::MINUI:
      kind_str = "'minui";
      break;
    case vector::CombiningKind::MINSI:
      kind_str = "'minsi";
      break;
    case vector::CombiningKind::MAXUI:
      kind_str = "'maxui";
      break;
    case vector::CombiningKind::MAXSI:
      kind_str = "'maxsi";
      break;
    case vector::CombiningKind::AND:
      kind_str = "'and";
      break;
    case vector::CombiningKind::OR:
      kind_str = "'or";
      break;
    case vector::CombiningKind::XOR:
      kind_str = "'xor";
      break;
    default:
      kind_str = "'nop";
    }
    std::string ret_str =
        "(vector:reduction (" + MLIRValVisit(src) + " " + kind_str + ") )";
    return ret_str;
  }

  std::string visit(vector::ReshapeOp reshapeOp) { return ""; }

  std::string visit(vector::SplatOp splatOp) {
    VectorType splatType = splatOp.getType();
    unsigned rank = splatType.getRank();
    std::string source = MLIRValVisit(splatOp.getInput());
    std::string ret_str =
        "(vector:splat (" + source + " " + std::to_string(rank) + ") )";
    return ret_str;
  }

  std::string visit(vector::TransposeOp transposeOp) { return ""; }

  std::string visit(vector::LoadOp loadOp) {
    bool is_sca = isa<IntegerType>(loadOp->getResult(0).getType());

    if (LoadToRegMap.find(loadOp) != LoadToRegMap.end()) {
      return "reg_" + std::to_string(LoadToRegMap[loadOp]);
    }

    if (is_sca) {
      auto intType = dyn_cast<VectorType>(loadOp->getResult(0).getType());
      std::string bits = std::to_string(intType.getElementTypeBitWidth() * 1);
      unsigned reg_counter = RegToLoadMap.size() + RegToTransferReadMap.size() +
                             RegToVariableMap.size();
      std::string reg_name = "reg_" + std::to_string(reg_counter);
      RegToLoadMap[reg_counter] = loadOp;
      LoadToRegMap[loadOp] = reg_counter;
      return reg_name;
    } else {

      auto vecType = dyn_cast<VectorType>(loadOp->getResult(0).getType());
      std::string bits = std::to_string(vecType.getElementTypeBitWidth() *
                                        vecType.getNumElements());
      unsigned reg_counter = RegToLoadMap.size() + RegToTransferReadMap.size() +
                             RegToVariableMap.size();
      std::string reg_name = "reg_" + std::to_string(reg_counter);
      RegToLoadMap[reg_counter] = loadOp;
      LoadToRegMap[loadOp] = reg_counter;
      // std::string load_buff = define_load_buffer(op);
      return reg_name; //+ load_buff + "\n"+ "(load " + reg_name
                       //+ " " + rkt_idx + " " + alignment + ")";
      // return tabs() + "(load " + op->name + " " + rkt_idx + " " +
      // alignment
      // + ")";
    }
  }

  std::string visit(vector::TransferReadOp transferReadOp) {

    bool is_sca = isa<IntegerType>(transferReadOp->getResult(0).getType());
    if (TransferReadToRegMap.find(transferReadOp) !=
        TransferReadToRegMap.end()) {
      return "reg_" + std::to_string(TransferReadToRegMap[transferReadOp]);
    }

    if (is_sca) {
      auto intType =
          dyn_cast<VectorType>(transferReadOp->getResult(0).getType());
      std::string bits = std::to_string(intType.getElementTypeBitWidth() * 1);
      unsigned reg_counter = RegToLoadMap.size() + RegToTransferReadMap.size() +
                             RegToVariableMap.size();
      std::string reg_name = "reg_" + std::to_string(reg_counter);
      RegToTransferReadMap[reg_counter] = transferReadOp;
      TransferReadToRegMap[transferReadOp] = reg_counter;
      return reg_name;
    } else {

      auto vecType =
          dyn_cast<VectorType>(transferReadOp->getResult(0).getType());
      std::string bits = std::to_string(vecType.getElementTypeBitWidth() *
                                        vecType.getNumElements());
      unsigned reg_counter = RegToLoadMap.size() + RegToTransferReadMap.size() +
                             RegToVariableMap.size();
      std::string reg_name = "reg_" + std::to_string(reg_counter);
      RegToTransferReadMap[reg_counter] = transferReadOp;
      TransferReadToRegMap[transferReadOp] = reg_counter;

      return reg_name;
    }
  }
};
} // namespace

void HydrideArithPass::runOnOperation() {

  LLVMConversionTarget target(getContext());
  target.addLegalOp<gpu::GPUModuleOp>();
  // target.addLegalDialect<mlir::LLVMDialect>();
  LLVMTypeConverter typeConverter(&getContext());
  ModuleOp mainModule = getOperation();
  unsigned expr_id = 0;
  std::string curr_func_name = benchmark_name;
  // curr_func_name = std::string();
  mainModule->walk([&](gpu::GPUModuleOp mainGPUModule) {
    DBGS() << "*** HydrideArithPass Running on " << mainGPUModule.getName()
           << "\n";
    // Compute the operation statistics for the currently visited operation.
    mainGPUModule->walk([&](gpu::GPUFuncOp funcOp) {
      DBGS() << "curr_func_name: " << curr_func_name << "\n";
      funcOp.walk([&](Operation *op) {
        if (auto returnOp = dyn_cast<gpu::ReturnOp>(op)) {
          RootExprOp.push(op);
          DBGS() << "Insert" << *op << "as gpu::ReturnOp\n";
        }
        if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
          RootExprOp.push(op);
          DBGS() << "Insert" << *op << "as vector::StoreOp\n";
        }
        if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
          RootExprOp.push(op);
          DBGS() << "Insert" << *op << "as vector::TransferWriteOp\n";
        }
        if (auto scfYieldOp = dyn_cast<scf::YieldOp>(op)) {
          RootExprOp.push(op);
          DBGS() << "Insert" << *op << "as scf::YieldOp\n";
        }
        /* if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
          RootExprOp.push(op);
        } */
        if (op->getNumResults() > 0) {
          auto vecType = dyn_cast<VectorType>(op->getResult(0).getType());
          if (vecType && vecType.getElementType()) {
            op_lanes = vecType.getNumElements();
          }
        }
      });
    });

    // move to own function
    // EmitSynthesis();

    while (!RootExprOp.empty()) {

      auto op = RootExprOp.front();
      // if result, expr from MLIRValVisit for all results
      // if store, expr from MLIRValVisit on single source vector
      // if other op, expr from MLIRValVisit for all operands
      std::string expr;
      // if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      //   for (Value operand : op->getOperands()) {
      //     expr = MLIRValVisit(operand);
      //     EmitSynthesis(curr_func_name, expr, expr_id);
      //     std::vector<Type> arg_types(RegToLoadMap.size() +
      //                                 RegToTransferReadMap.size() +
      //                                 RegToVariableMap.size());
      //     std::vector<Value> args_vec(RegToLoadMap.size() +
      //                                 RegToTransferReadMap.size() +
      //                                 RegToVariableMap.size());

      //     for (auto reg : RegToLoadMap) {
      //       arg_types[reg.first] = reg.second.getType();
      //       args_vec[reg.first] = reg.second;
      //     }

      //     for (auto reg : RegToTransferReadMap) {
      //       arg_types[reg.first] = reg.second.getVector().getType();
      //       args_vec[reg.first] = reg.second;
      //     }

      //     for (auto reg : RegToVariableMap) {
      //       auto val = Value::getFromOpaquePointer(reg.second);
      //       arg_types[reg.first] = val.getType();
      //       args_vec[reg.first] = val;
      //     }

      //     MLIRContext *ctx = returnOp.getContext();
      //     MyPatternRewriter rewriter(ctx);
      //     FlatSymbolRefAttr sym_ref = getOrInsertHydrideFunc(
      //         rewriter, mainGPUModule, expr_id, curr_func_name,
      //         operand.getType(), arg_types);
      //     rewriter.setInsertionPoint(returnOp);
      //     Location loc = returnOp->getLoc();
      //     auto callOp = rewriter.create<LLVM::CallOp>(
      //         loc, TypeRange(operand.getType()), sym_ref,
      //         ValueRange(args_vec));
      //     operand.replaceAllUsesWith(callOp.getResult());
      //     expr_id++;
      //   }
      // }
      // if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
      //   auto valToStore = storeOp.getValueToStore();
      //   auto valToStoreType = valToStore.getType();
      //   expr = MLIRValVisit(valToStore);
      //   EmitSynthesis(curr_func_name, expr, expr_id);

      //   std::vector<Type> arg_types(RegToLoadMap.size() +
      //                               RegToTransferReadMap.size() +
      //                               RegToVariableMap.size());
      //   std::vector<Value> args_vec(RegToLoadMap.size() +
      //                               RegToTransferReadMap.size() +
      //                               RegToVariableMap.size());
      //   for (auto reg : RegToLoadMap) {
      //     arg_types[reg.first] = reg.second.getType();
      //     args_vec[reg.first] = reg.second;
      //   }

      //   for (auto reg : RegToVariableMap) {
      //     auto val = Value::getFromOpaquePointer(reg.second);
      //     arg_types[reg.first] = val.getType();
      //     args_vec[reg.first] = val;
      //   }

      //   MLIRContext *ctx = storeOp.getContext();
      //   MyPatternRewriter rewriter(ctx);
      //   FlatSymbolRefAttr sym_ref =
      //       getOrInsertHydrideFunc(rewriter, mainGPUModule, expr_id,
      //                              curr_func_name, valToStoreType,
      //                              arg_types);
      //   // get all args in
      //   rewriter.setInsertionPoint(storeOp);
      //   Location loc = storeOp->getLoc();
      //   auto callOp = rewriter.create<LLVM::CallOp>(
      //       loc, TypeRange(valToStoreType), sym_ref, ValueRange(args_vec));
      //   storeOp->setOperands(0, 1, callOp.getResult());

      //   expr_id++;
      // }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        auto valToStore = yieldOp->getOperand(0);
        auto valToStoreType = valToStore.getType();
        expr = MLIRValVisit(valToStore);
        DBGS() << "Emit: " << expr << " recursively...\n";
        EmitSynthesis(curr_func_name, expr, expr_id);
        std::vector<Type> arg_types(RegToLoadMap.size() +
                                    RegToTransferReadMap.size() +
                                    RegToVariableMap.size());
        std::vector<Value> args_vec(RegToLoadMap.size() +
                                    RegToTransferReadMap.size() +
                                    RegToVariableMap.size());

        for (auto reg : RegToTransferReadMap) {
          arg_types[reg.first] = reg.second.getVector().getType();
          args_vec[reg.first] = reg.second;
        }

        for (auto reg : RegToVariableMap) {
          auto val = Value::getFromOpaquePointer(reg.second);
          arg_types[reg.first] = val.getType();
          args_vec[reg.first] = val;
        }

        MLIRContext *ctx = yieldOp.getContext();
        MyPatternRewriter rewriter(ctx);

        FlatSymbolRefAttr sym_ref =
            getOrInsertHydrideFunc(rewriter, mainGPUModule, expr_id,
                                   curr_func_name, valToStoreType, arg_types);
        // get all args in
        rewriter.setInsertionPoint(yieldOp);
        Location loc = yieldOp->getLoc();
        auto callOp = rewriter.create<spirv::FunctionCallOp>(
            loc, TypeRange(valToStoreType), sym_ref, ValueRange(args_vec));
        yieldOp->setOperands(0, 1, callOp.getResult(0));
        expr_id++;
      } else if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
        auto valToStore = transferWriteOp.getVector();
        auto valToStoreType = valToStore.getType();
        expr = MLIRValVisit(valToStore);
        DBGS() << "Emit: " << expr << " recursively...\n";
        EmitSynthesis(curr_func_name, expr, expr_id);
        std::vector<Type> arg_types(RegToLoadMap.size() +
                                    RegToTransferReadMap.size() +
                                    RegToVariableMap.size());
        std::vector<Value> args_vec(RegToLoadMap.size() +
                                    RegToTransferReadMap.size() +
                                    RegToVariableMap.size());

        for (auto reg : RegToTransferReadMap) {
          arg_types[reg.first] = reg.second.getVector().getType();
          args_vec[reg.first] = reg.second;
        }

        for (auto reg : RegToVariableMap) {
          auto val = Value::getFromOpaquePointer(reg.second);
          arg_types[reg.first] = val.getType();
          args_vec[reg.first] = val;
        }

        MLIRContext *ctx = transferWriteOp.getContext();
        MyPatternRewriter rewriter(ctx);

        FlatSymbolRefAttr sym_ref =
            getOrInsertHydrideFunc(rewriter, mainGPUModule, expr_id,
                                   curr_func_name, valToStoreType, arg_types);
        // get all args in
        rewriter.setInsertionPoint(transferWriteOp);
        Location loc = transferWriteOp->getLoc();
        auto callOp = rewriter.create<spirv::FunctionCallOp>(
            loc, TypeRange(valToStoreType), sym_ref, ValueRange(args_vec));
        transferWriteOp->setOperands(0, 1, callOp.getResult(0));
        expr_id++;
      }

      else {

        // if (op->getNumResults() > 0) {
        //   if (op->getDialect()->getNamespace() == "arith") {
        //     expr = MLIRArithOpVisit(op);
        //   } else if (op->getDialect()->getNamespace() == "vector") {
        //     expr = MLIRVectorOpVisit(op);
        //   } else
        //     continue;
        //   EmitSynthesis(curr_func_name, expr, expr_id);
        //   expr_id++;
        // }
      }

      RegToVariableMap.clear();
      VariableToRegMap.clear();
      RegToTransferReadMap.clear();
      TransferReadToRegMap.clear();
      RegToLoadMap.clear();
      LoadToRegMap.clear();
      RootExprOp.pop();
    }
    DBGS() << *mainGPUModule << "\n";
  });
  DBGS() << "--------------------------------\n";
  DBGS() << "Hydride Synthesis Complete\n";
  EmitLLVMCode();
  DBGS() << *mainModule << "\n";
}

FlatSymbolRefAttr HydrideArithPass::getOrInsertHydrideFunc(
    MyPatternRewriter &rewriter, gpu::GPUModuleOp module, int expr_id,
    std::string curr_func_name, Type retType, std::vector<Type> argTypes) {
  // add logic for naming
  std::string func_name =
      "hydride.node." + curr_func_name + "." + std::to_string(expr_id);
  auto *context = module.getContext();

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  // auto llvmI32Ty = IntegerType::get(context, 32);
  // auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context,
  // 8));

  auto spirvFnType = FunctionType::get(context, argTypes, retType);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto newFuncOp =
      rewriter.create<spirv::FuncOp>(module.getLoc(), func_name, spirvFnType);

  newFuncOp->setAttr("VectorComputeFunctionINTEL", rewriter.getUnitAttr());
  auto linkageTypeAttr =
      rewriter.getAttr<spirv::LinkageTypeAttr>(spirv::LinkageType::Import);
  auto nameAttr = StringAttr::get(rewriter.getContext(), func_name);
  auto linkage = spirv::LinkageAttributesAttr::get(rewriter.getContext(),
                                                   nameAttr, linkageTypeAttr);
  newFuncOp.setLinkageAttributesAttr(linkage);
  return SymbolRefAttr::get(context, func_name);
}

std::string HydrideArithPass::MLIRValVisit(Value val) {
  DBGS() << "Visiting: " << val << "\n";
  void *valAsOpaquePointer = val.getAsOpaquePointer();
  auto valDefiningOp = val.getDefiningOp();
  if (valDefiningOp != NULL) {
    if (valDefiningOp->getDialect() != NULL) {
      if (valDefiningOp->getDialect()->getNamespace() == "arith") {
        return MLIRArithOpVisit(valDefiningOp);
      } else if (valDefiningOp->getDialect()->getNamespace() == "vector") {
        return MLIRVectorOpVisit(valDefiningOp);
      } else if (valDefiningOp->getDialect()->getNamespace() == "xegpu") {
        // TODO:?
      } else {
        DBGS() << "Unknown Op" << valDefiningOp;
        if (VariableToRegMap.find(valDefiningOp) != VariableToRegMap.end()) {
          std::string reg_name =
              "reg_" + std::to_string(VariableToRegMap[valDefiningOp]);
          return reg_name;
        }

        unsigned reg_counter =
            (RegToLoadMap.size() + RegToTransferReadMap.size() +
             RegToVariableMap.size());

        std::string reg_name = "reg_" + std::to_string(reg_counter);

        RegToVariableMap[reg_counter] = valDefiningOp;
        VariableToRegMap[valDefiningOp] = reg_counter;

        return reg_name; // op->name;
      }
    }
  }

  if (isa<BlockArgument>(val)) {

    if (VariableToRegMap.find(valAsOpaquePointer) != VariableToRegMap.end()) {
      std::string reg_name =
          "reg_" + std::to_string(VariableToRegMap[valAsOpaquePointer]);
      return reg_name;
    }

    unsigned reg_counter = (RegToLoadMap.size() + RegToTransferReadMap.size() +
                            RegToVariableMap.size());

    std::string reg_name = "reg_" + std::to_string(reg_counter);

    RegToVariableMap[reg_counter] = valAsOpaquePointer;
    VariableToRegMap[valAsOpaquePointer] = reg_counter;

    DBGS() << "Assigning: " << reg_name << " to " << val << "\n";
    return reg_name; // op->name;
  }

  return "";
}

std::string HydrideArithPass::print_binary_op(std::string bv_name, Value a,
                                              Value b, bool is_vector_op) {

  if (is_vector_op) {

    // indent.push(indent.top() + 1);
    std::string rkt_lhs = MLIRValVisit(a);
    std::string rkt_rhs = MLIRValVisit(b);

    return "(arith:tensor-" + bv_name + " " + rkt_lhs + " " + rkt_rhs + ")";
  } else {

    std::string rkt_lhs = MLIRValVisit(a);
    std::string rkt_rhs = MLIRValVisit(b);
    return "(arith:sca-" + bv_name + " " + rkt_lhs + " " + rkt_rhs + ")";
  }
}

std::string HydrideArithPass::MLIRArithOpVisit(Operation *op) {

  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    return visit(constantOp);
  }

  if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
    return visit(addOp);
  }

  if (auto divUIOp = dyn_cast<arith::DivUIOp>(op)) {
    return visit(divUIOp);
  }

  if (auto divSIOp = dyn_cast<arith::DivSIOp>(op)) {
    return visit(divSIOp);
  }

  if (auto subOp = dyn_cast<arith::SubIOp>(op)) {
    return visit(subOp);
  }

  if (auto divUIOp = dyn_cast<arith::DivUIOp>(op)) {
    return visit(divUIOp);
  }

  if (auto minUIOp = dyn_cast<arith::MinUIOp>(op)) {
    return visit(minUIOp);
  }

  if (auto minSIOp = dyn_cast<arith::MinSIOp>(op)) {
    return visit(minSIOp);
  }

  if (auto maxUIOp = dyn_cast<arith::MaxUIOp>(op)) {
    return visit(maxUIOp);
  }

  if (auto maxSIOp = dyn_cast<arith::MaxSIOp>(op)) {
    return visit(maxSIOp);
  }

  if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
    return visit(mulOp);
  }

  if (auto shrUIOp = dyn_cast<arith::ShRUIOp>(op)) {
    return visit(shrUIOp);
  }

  if (auto shrSIOp = dyn_cast<arith::ShRSIOp>(op)) {
    return visit(shrSIOp);
  }

  if (auto shlIOp = dyn_cast<arith::ShLIOp>(op)) {
    return visit(shlIOp);
  }

  if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
    return visit(mulOp);
  }

  if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
    return visit(cmpOp);
  }

  if (auto extOp = dyn_cast<arith::ExtSIOp>(op)) {
    return visit(extOp);
  }

  for (Value operand : op->getOperands()) {
    if (auto OpOperandRoot = operand.getDefiningOp()) {
      RootExprOp.push(OpOperandRoot);
    }
  }

  return "";
}

std::string HydrideArithPass::MLIRVectorOpVisit(Operation *op) {

  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return visit(loadOp);
  }
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op)) {
    return visit(transferReadOp);
  }

  if (auto bitcastOp = dyn_cast<vector::BitCastOp>(op)) {
    return visit(bitcastOp);
  }

  if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(op)) {
    return visit(broadcastOp);
  }

  if (auto extractElementOp = dyn_cast<vector::ExtractElementOp>(op)) {
    return visit(extractElementOp);
  }

  if (auto extractOp = dyn_cast<vector::ExtractOp>(op)) {
    return visit(extractOp);
  }

  if (auto extractStridedSliceOp =
          dyn_cast<vector::ExtractStridedSliceOp>(op)) {
    return visit(extractStridedSliceOp);
  }

  if (auto insertStridedSliceOp = dyn_cast<vector::InsertStridedSliceOp>(op)) {
    return visit(insertStridedSliceOp);
  }

  if (auto fmaOp = dyn_cast<vector::FMAOp>(op)) {
    return visit(fmaOp);
  }

  if (auto flatTransposeOp = dyn_cast<vector::FlatTransposeOp>(op)) {
    return visit(flatTransposeOp);
  }

  if (auto matMulOp = dyn_cast<vector::MatmulOp>(op)) {
    return visit(matMulOp);
  }

  if (auto reductionOp = dyn_cast<vector::ReductionOp>(op)) {
    return visit(reductionOp);
  }

  if (auto splatOp = dyn_cast<vector::SplatOp>(op)) {
    return visit(splatOp);
  }

  if (auto transposeOp = dyn_cast<vector::TransposeOp>(op)) {
    return visit(transposeOp);
  }

  for (Value operand : op->getOperands()) {
    if (auto OpOperandRoot = operand.getDefiningOp()) {
      if (!isa<vector::LoadOp>(OpOperandRoot) &&
          !isa<vector::LoadOp>(OpOperandRoot)) {
        RootExprOp.push(OpOperandRoot);
      }

      if (!isa<vector::TransferReadOp>(OpOperandRoot) &&
          !isa<vector::TransferReadOp>(OpOperandRoot)) {
        RootExprOp.push(OpOperandRoot);
      }
    }
  }
  auto val = op->getResult(0);
  void *valAsOpaquePointer = op->getResult(0).getAsOpaquePointer();
  if (VariableToRegMap.find(valAsOpaquePointer) != VariableToRegMap.end()) {
    std::string reg_name =
        "reg_" + std::to_string(VariableToRegMap[valAsOpaquePointer]);
    return reg_name;
  }

  unsigned reg_counter = (RegToLoadMap.size() + RegToTransferReadMap.size() +
                          RegToVariableMap.size());

  std::string reg_name = "reg_" + std::to_string(reg_counter);

  RegToVariableMap[reg_counter] = valAsOpaquePointer;
  VariableToRegMap[valAsOpaquePointer] = reg_counter;

  DBGS() << "Assigning: " << reg_name << " to " << val << "\n";
  return reg_name; // op->name;
  // return "";
}

void HydrideArithPass::EmitLLVMCode() {
  const char *benchmark_name = getenv("HYDRIDE_BENCHMARK");
  assert(benchmark_name && "HYDRIDE_BENCHMARK environment variable must be "
                           "defined for hydride llvm code-gen");

  const char *hydride_src = getenv("HYDRIDE_ROOT");
  assert(hydride_src &&
         "HYDRIDE_ROOT environment variable needs to be defined for codegen");

  const char *legalizer_so = getenv("LEGALIZER_PATH");
  assert(legalizer_so && "LEGALIZER_PATH environment variable must be defined "
                         "for hydride codegen");

  const char *intrin_wrapper = getenv("INTRINSICS_LL");
  assert(intrin_wrapper &&
         "INTRINSICS_LL must be defined for hydride llvm code-gen");

  std::string codegen_script_path =
      std::string(hydride_src) +
      "/codegen-generator/tools/low-level-codegen/RoseLowLevelCodeGen.py";
  std::string input_file = "/tmp/" + std::string(benchmark_name) + ".rkt";
  std::string output_file = "/tmp/" + std::string(benchmark_name) + ".ll";
  // python3
  // $HYDRIDE_ROOT/codegen-generator/tools/low-level-codegen/RoseLowLevelCodeGen.py
  // /tmp/forward_kernel.rkt
  // $HYDRIDE_ROOT/codegen-generator/tools/low-level-codegen/InstSelectors/visa/build/libVISALegalizer.so
  // $HYDRIDE_ROOT/codegen-generator/tools/low-level-codegen/InstSelectors/visa/visa_wrapper.ll
  // -visa-hydride-legalize /tmp/forward_kernel.ll
  std::string cmd = "HYDRIDE_VISA_FLAG=1 python3 " + codegen_script_path + " " +
                    input_file + " " + std::string(legalizer_so) + " " +
                    std::string(intrin_wrapper) + " -visa-hydride-legalize " +
                    output_file;
  DBGS() << "About to execute " << cmd << "\n";
  if (!DisableLegalize) {
    auto start = std::chrono::system_clock::now();

    int ret_code = system(cmd.c_str());

    assert(ret_code == 0 && "Codegeneration crashed, exiting ...");

    auto end = std::chrono::system_clock::now();
    DBGS() << "Compilation completed with return code:\t" << ret_code << "\n";

    std::chrono::duration<double> elapsed_seconds = end - start;

    DBGS() << "Compilation took " << elapsed_seconds.count() << " seconds ..."
           << "\n";
  }
}

void HydrideArithPass::EmitSynthesis(std::string benchmark_name,
                                     std::string expr, unsigned expr_id) {
  HydrideSynthEmitter HSE(benchmark_name, LoadToRegMap, VariableToRegMap,
                          TransferReadToRegMap);
  std::string errorMessage;
  std::string out_str;

  std::string outputFilename = std::string("mlir_expr_") + benchmark_name +
                               std::to_string(expr_id) + ".rkt";

  auto file = openOutputFile(outputFilename, &errorMessage);
  if (!file) {
    DBGS() << errorMessage << "\n";
    return;
  }

  out_str += HSE.emit_racket_imports() + "\n";
  out_str += HSE.emit_set_current_bitwidth() += "\n";
  out_str += HSE.emit_set_memory_limit(20000) += "\n";
  out_str += HSE.emit_symbolic_buffers() += "\n";
  out_str += HSE.emit_buffer_id_map("id-map") += "\n\n";

  out_str += "(define mlir-expr \n";
  out_str += expr + "\n";
  out_str += ")\n\n";

  out_str += "(clear-vc!)\n";

  std::string prev_hash_path = HSE.get_synthlog_hash_filepath(expr_id - 1);
  std::string prev_hash_name = HSE.get_synthlog_hash_name(expr_id - 1);

  const char *expr_depth_var = getenv("HL_EXPR_DEPTH");

  int expr_depth = 2;

  if (expr_depth_var) {
    expr_depth = (*expr_depth_var) - '0';
  }

  // std::string synth_target_str = "x86";
  std::string synth_target_str = "x86";

  if (!synth_target.empty()) {
    synth_target_str = synth_target;
  }

  out_str += "(define synth-res " +
             HSE.emit_hydride_synthesis(
                 "mlir-expr", /* expr depth */ expr_depth,
                 /* VF*/ op_lanes, /* Reg Hash map name */ "id-map",
                 /* Previous hash file path */ prev_hash_path,
                 /* Previous hash  name */ prev_hash_name,
                 /* target id */ synth_target_str) +
             ")" + "\n";
  out_str += "(dump-synth-res-with-typeinfo synth-res id-map)\n";

  std::string fn_name =
      "hydride.node." + benchmark_name + "." + std::to_string(expr_id);
  out_str +=
      HSE.emit_compile_to_llvm("synth-res", "id-map", fn_name, benchmark_name);

  std::string cur_hash_path = HSE.get_synthlog_hash_filepath(expr_id);
  std::string cur_hash_name = HSE.get_synthlog_hash_name(expr_id);

  out_str +=
      HSE.emit_write_synth_log_to_file("/tmp/" + cur_hash_path, cur_hash_name);

  file->os() << out_str;
  file->keep();
  // close file
  file->os().close();

  std::string cmd = "racket " + outputFilename;
  DBGS() << "About to execute " << cmd << "\n";
  if (!DisableSynth) {
    auto start = std::chrono::system_clock::now();
    int ret_code = system(cmd.c_str());
    auto end = std::chrono::system_clock::now();
    assert(ret_code == 0 && "Synthesis crashed, exiting ...");
    DBGS() << "Synthesis completed with return code:\t" << ret_code << "\n";

    std::chrono::duration<double> elapsed_seconds = end - start;
    DBGS() << "Synthesis took " << elapsed_seconds.count() << "seconds ..."
           << "\n";
  }

  expr_id++;
  op_lanes = 0;
}

namespace imex {
std::unique_ptr<mlir::Pass> createHydrideArithPass() {
  return std::make_unique<HydrideArithPass>();
}

std::unique_ptr<mlir::Pass> createHydrideArithPass(std::string synth_target) {
  return std::make_unique<HydrideArithPass>(synth_target);
}
} // namespace imex