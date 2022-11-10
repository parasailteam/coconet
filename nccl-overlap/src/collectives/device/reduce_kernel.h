/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_
#include <limits>
#include <stdio.h>
#include <assert.h>
#include <curand_kernel.h>
template<typename T>
struct FuncNull {
  __device__ T operator()(const T x, const T y) const {
    return 0;
  }
};

template<typename T>
struct FuncSum {
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<typename T> 
struct FuncSub {
  __device__ T operator()(const T x, const T y) const {
    return x - y;
  }
};

template<typename T> 
struct FuncSqrt {
  __device__ T operator()(const T x) const {
    return (T)sqrt((double)x);
  }
};

template<> 
struct FuncSqrt<half> {
  __device__ half2 operator()(const half2 x) const {
    assert(false);
    return x;
  }

  __device__ half operator()(const half x) const {
    assert(false);
    return x;
  }
};

template<typename T>    struct r {    
  __device__ T operator()(const T w, const T m, const T v) {
    T out = (m) / ((T)sqrtf((float)v) + (T)1e-6) + ((T)1.0f) * w;
    // float fout = ((float)out);
    // if (abs((fout-(-1.0f))/fout) >= 1e-4) {
    //   printf("out %f S4 %f\n", (float)out, (float)S4);
    // }
   return out;
}};

template<typename T>    struct LAMBWeightUpdate {    
  __device__ T operator()(const T w, const T ratio, const T rLambdaWeight) {
    T out = w - ratio * rLambdaWeight;
    // float fout = ((float)out);
    // if (abs((fout-(-1.0f))/fout) >= 1e-4) {
    //   printf("out %f S4 %f\n", (float)out, (float)S4);
    // }
    // if (threadIdx.x == 0) {
    //   printf("out '%f' w '%f' ratio %f rLambdaWeight '%f'\n", (float)out, (float)w, (float)ratio, (float)rLambdaWeight);
    // }
   return out;
}};


template<>
struct FuncSub<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x + fy.x;
    fr.y = fx.y + fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd(x, y);
#else
    return __float2half( __half2float(x) + __half2float(y) );
#endif
  }
};

template<typename T>
struct FuncProd {
  __device__ T operator()(const T x, const T y) const {
    return x * y;
  }
};

template<typename T>
struct FuncDiv {
  __device__ T operator()(const T x, const T y) const {
    return x / y;
  }
};

template<typename T>
struct FuncPrint {
  __device__ T operator()(const T x) const {
    //if(threadIdx.x + blockDim.x*blockIdx.x==0) 
    printf("x %f\n", (float)x);
    return 0;
  }
};

template<typename T>
struct FuncEq {
  __device__ T operator()(const T x) const {
    //if(threadIdx.x + blockDim.x*blockIdx.x==0) 
    if(abs((x - 4.0f)/x) <= 1e-4) {
      printf("x is 4.0\n");
    }
    return 0;
  }
};

template<typename T>
struct FuncPow {
  __device__ T operator()(const T x, const T y) const {
    return (T)powf((float)x, (float)y);
  }
};

template<typename T>
struct FuncFirstMomentUpdate {
  //m[t] = beta1 * m[t-1] + (1-beta1)*g
  __device__ T operator()(const T m, const T grad, const T beta) {
    return beta * m + (1-beta) * grad;
  }
};

template<>
struct FuncFirstMomentUpdate<half> {
  __device__ half operator()(const half m, const half grad, const half beta1) {
    assert(false);
    return m; //return grad * m + (1-beta1) * grad;
  }
};

template<>
struct FuncFirstMomentUpdate<half2> {
  __device__ half2 operator()(const half2 m, const half2 grad, const half2 beta1) {
    assert(false);
    return m; //return grad * m + (1-beta1) * grad;
  }
};

template<typename T>
struct FuncSecondMomentUpdate {
  //v[t] = beta2 * v[t-1] + (1-beta2)*g*g
  __device__ T operator()(const T m, const T grad, const T beta) {
    return beta * m + (1-beta) * grad * grad;
  }
};

template<>
struct FuncSecondMomentUpdate<half> {
  __device__ half operator()(const half m, const half grad, const half beta1) {
    assert(false);
    return m; //return grad * m + (1-beta1) * grad;
  }
};

template<>
struct FuncSecondMomentUpdate<half2> {
  __device__ half2 operator()(const half2 m, const half2 grad, const half2 beta1) {
    assert(false);
    return m; //return grad * m + (1-beta1) * grad;
  }
};


template<typename T>
struct FuncDropout {
  __device__ T operator()(const T val, const T addTensorVal, const T biasVal, curandState* randState, const float p) {
    return (curand_uniform(randState) < p ? val  : (T)0.0f) + biasVal + addTensorVal;
  }
};

template<>
struct FuncDropout<half> {
  __device__ half2 operator()(const half2 val, const half2 addTensorVal, const half2 biasVal,curandState* randState, const float p) {
    half2 v = (curand_uniform(randState) < p ? val  : __float2half2_rn(0.0f));
    return __hadd2_sat(__hadd2_sat(v, biasVal), addTensorVal);
  }
};

template<>
struct FuncDropout<half2> {
  __device__ half2 operator()(const half2 val, const half2 addTensorVal, const half2 biasVal,curandState* randState, const float p) {
    half2 v = (curand_uniform(randState) < p ? val  : __float2half2_rn(0.0f));
    return __hadd2_sat(__hadd2_sat(v, biasVal), addTensorVal);
  }
};

template<typename T>
struct FuncBiasCorrection {
  //m_[t] = m[t]/(1-beta1^t)
  //v_[t] = v[t]/(1-beta2^t)
  __device__ T operator()(const T moment, const T beta, int t) {
    return moment/(1-FuncPow<T>()(beta, (T)t));
  }
};

template<>
struct FuncBiasCorrection<half> {
  __device__ half operator()(const half moment, const half beta, int t) {
    assert(false);
    return moment;//return moment/(1-FuncPow<T>()(beta, t));
  }
};

template<>
struct FuncBiasCorrection<half2> {
  __device__ half2 operator()(const half2 moment, const half2 beta, int t) {
    assert(false);
    return moment;//return moment/(1-FuncPow<T>()(beta, t));
  }
};

template<typename T>
struct FuncAdamWeightUpdate {
  //w[t] = w[t-1] - alpha*m_[t]/(sqrt(v_[t]) + epsilon)
  __device__ T operator()(const T weight, const T m, const T v, const T alpha, const T epsilon) {
    return weight + alpha * m/(FuncSqrt<T>()(v) + epsilon);
  }
};


template<>
struct FuncAdamWeightUpdate<half2> {
  __device__ half2 operator()(const half2 weight, const half2 m, const half2 v, const half2 alpha, const half2 epsilon) {
    assert(false);
    return weight;
    //return weight - alpha * m/(FuncSqrt<T>(v) + epsilon);
  }
};

template<>
struct FuncAdamWeightUpdate<half> {
  __device__ half operator()(const half weight, const half m, const half v, const half alpha, const half epsilon) {
    assert(false);
    return weight;
  }
};

#include "common_kernel.h"

template<typename T>
struct FuncMax {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? y : x;
  }
};

template<typename T>
struct FuncMin {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? x : y;
  }
};

#define MASK0 0x00ff00ff
#define MASK1 0xff00ff00
static __device__ uint32_t addChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit addition */
  const uint32_t x0 = x & MASK0;
  const uint32_t x1 = x & MASK1;
  const uint32_t y0 = y & MASK0;
  const uint32_t y1 = y & MASK1;
  const uint32_t r0 = (x0+y0);
  const uint32_t r1 = (x1+y1);
  return (r0 & MASK0) | (r1 & MASK1);
}

template<>
struct FuncSum<int8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vadd4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x+y;
  }
};
template<>
struct FuncSum<uint8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vadd4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x+y;
  }
};

static __device__ uint32_t mulChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit multiplication */
  union converter { uint32_t storage; char4 a; };
  converter cx, cy, cr;
  cx.storage = x;
  cy.storage = y;
  cr.a.x = cx.a.x * cy.a.x;
  cr.a.y = cx.a.y * cy.a.y;
  cr.a.z = cx.a.z * cy.a.z;
  cr.a.w = cx.a.w * cy.a.w;
  return cr.storage;
}

template<>
struct FuncProd<int8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x*y;
  }
};
template<>
struct FuncProd<uint8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x*y;
  }
};

template<>
struct FuncMax<int8_t> {
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmax4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x>y) ? x : y;
  }
};
template<>
struct FuncMax<uint8_t> {
  union converter { uint32_t storage; uchar4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x>y) ? x : y;
  }
};

template<>
struct FuncMin<int8_t> {
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmin4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x<y) ? x : y;
  }
};
template<>
struct FuncMin<uint8_t> {
  union converter { uint32_t storage; uchar4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmin4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x<y) ? x : y;
  }
};

template<>
struct FuncSum<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x + fy.x;
    fr.y = fx.y + fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd(x, y);
#else
    return __float2half( __half2float(x) + __half2float(y) );
#endif
  }
};

template<>
struct FuncProd<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x * fy.x;
    fr.y = fx.y * fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul(x, y);
#else
    return __float2half( __half2float(x) * __half2float(y) );
#endif
  }
};

template<>
struct FuncMax<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fmaxf(fx.x, fy.x);
    fr.y = fmaxf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fmaxf(fx, fy);
    return __float2half(fm);
  }
};

template<>
struct FuncMin<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fminf(fx.x, fy.x);
    fr.y = fminf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fminf(fx, fy);
    return __float2half(fm);
  }
};
#endif // REDUCE_KERNEL_H_
