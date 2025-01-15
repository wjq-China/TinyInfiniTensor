#include "operators/matmul.h"
#include "core/tensor.h"
#include <algorithm>
#include <optional>

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
  IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
  std::ostringstream os;
  os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
     << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
     << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
     << "])";
  return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
  // =================================== 作业
  // ===================================
  // TODO：返回经过 matmul 操作后的 shape
  // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
  // =================================== 作业
  // =================================== return std::nullopt;
  auto A = inputs[0];
  auto B = inputs[1];
  if (transA) {
    auto shape = A->getDims();
    int rank = A->getRank();
    std::swap(shape[rank - 1], shape[rank - 2]);
    A->setShape(shape);
  }
  if (transB) {
    auto shape = B->getDims();
    int rank = B->getRank();
    std::swap(shape[rank - 1], shape[rank - 2]);
    B->setShape(shape);
  }
  // 广播除最后两维外的维度
  auto shapeA = A->getDims();
  auto shapeB = B->getDims();
  int rankA = A->getRank();
  int rankB = A->getRank();
  int rank = std::max(rankA, rankB);
  Shape shape(rank, 0);
  for (int i = 0; i < rank; i++) {
    int dimA = i < rankA ? shapeA[rankA - i - 1] : 1;
    int dimB = i < rankB ? shapeB[rankB - i - 1] : 1;
    if (i == 1) {
      shape[rank - i - 1] = shapeA[rankA - 2];
      continue;
    }
    if (i == 0) {
      shape[rank - i - 1] = shapeB[rankB - 1];
      continue;
    }
    if ((dimA == dimB) || (dimA == 1) | (dimB == 1)) {
      shape[rank - i - 1] = std::max(dimA, dimB);
    }
  }

  return {{shape}};
}

} // namespace infini