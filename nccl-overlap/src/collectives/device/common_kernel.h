/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include "reduce_kernel.h"
#include <cstdio>
#include <cstdint>

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>

#define Pp if(blockDim.x*blockIdx.x+threadIdx.x==0)printf("%s:%d\n",__FILE__,__LINE__);

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

typedef uint64_t PackType;

// unpack x and y to elements of type T and apply FUNC to each element
template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const T c1, const T c2) const;
  __device__ PackType operator()(const PackType x, T y, const int alpha) const;
  __device__ PackType operator()(const PackType x, const PackType y, const T alpha) const;
  __device__ PackType operator()(const PackType x, const PackType y) const;
  __device__ PackType operator()(const PackType x, const T alpha) const;
  __device__ PackType operator()(const PackType x) const;
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const;
  __device__ PackType LAMBWeightUpdate(const PackType w, T ratio, const PackType rLambdaWeight) const;
  __device__ PackType dropout(const PackType x, const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const;
};

struct FuncProd2 {
  template<typename T>
  __device__ T operator()(const T x, const T y) const {
    T z = x * y;
    return z;
  }
};

struct FuncSub2 {
  template<typename T> 
  __device__ T operator()(const T x, const T y) const {
    return x - y;
  }
};

template<typename T> 
struct FuncSum2 {
  __device__ T operator()(const T x, const T y) const {
    T z = x + y;
    return z;
  }
};


struct FuncFMA2 {
  template<typename T> 
  __device__ T operator()(const T x, const T y, const T a) const {
    T z = a * x + y;
    return z;
  }
};


template<class FUNC>
struct MULTI<FUNC, int8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const int8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const int8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const int8_t y, const int alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const int8_t c1, const int8_t c2) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{return 0;}
  __device__ PackType LAMBWeightUpdate(const PackType w, int8_t ratio, const PackType rLambdaWeight) const {return 0;}
  __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, uint8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

   __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const uint8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const uint8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const uint8_t y, const int alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const uint8_t c1, const uint8_t c2) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{return 0;}
  __device__ PackType LAMBWeightUpdate(const PackType w, uint8_t ratio, const PackType rLambdaWeight) const {return w;}
  __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, int32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(int32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      int32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const int32_t alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, alpha);
    cr.b = FUNC()(cx.b, alpha);

    return cr.storage;
  }

  
  __device__ PackType operator()(const PackType x, const PackType y, const int32_t alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const int32_t y, const int32_t alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, y, alpha);
    cr.b = FUNC()(cx.b, y, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const int32_t c1, const int32_t c2) {
    converter cx, cy, cz, cr;
    cx.storage = x;
    cy.storage = y;
    cz.storage = z;

    cr.a = FUNC()(cx.a, cy.a, cz.a, c1, c2);
    cr.b = FUNC()(cx.b, cy.b, cz.a, c1, c2);

    return cr.storage;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, int32_t ratio, const PackType rLambdaWeight) const {}
  __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, uint32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const uint32_t alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, alpha);
    cr.b = FUNC()(cx.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const uint32_t alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const int alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const uint32_t y, const int alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, y, alpha);
    cr.b = FUNC()(cx.b, y, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const uint32_t c1, const uint32_t c2) {
    converter cx, cy, cz, cr;
    cx.storage = x;
    cy.storage = y;
    cz.storage = z;

    cr.a = FUNC()(cx.a, cy.a, cz.a, c1, c2);
    cr.b = FUNC()(cx.b, cy.b, cz.a, c1, c2);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, int32_t ratio, const PackType rLambdaWeight) const {}
  __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, half> {
  static_assert(sizeof(PackType) == 4 * sizeof(half),
      "PackType must be four times the size of half.");

  struct PackHalf2 {
    half2 a, b;
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    #if 1
    //FuncSub do not work with half
    struct PackHalf2 cx, cy, cr;
    cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
    cy = *(reinterpret_cast<const struct PackHalf2*>(&y));

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return *(reinterpret_cast<PackType*>(&cr));
    #else
    assert(false);
    return x;
    #endif
  }

  __device__ PackType operator()(const PackType x, const half alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const half alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const half y, const int alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const half c1, const half c2) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, half ratio, const PackType rLambdaWeight) const {}
  __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const {
    struct PackHalf2 cx, cy, cz, cr;
    cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
    cy = *(reinterpret_cast<const struct PackHalf2*>(&addTensorVal));
    cz = *(reinterpret_cast<const struct PackHalf2*>(&biasVal));

    cr.a = FUNC()(cx.a, cy.a, cz.a, randState, val);
    cr.b = FUNC()(cx.b, cy.b, cz.b, randState, val);

    return *(reinterpret_cast<PackType*>(&cr));
  }
};

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const float alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, alpha);
    cr.b = FUNC()(cx.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const float alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
     converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a);
    cr.b = FUNC()(cx.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const float y, const int alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, y, alpha);
    cr.b = FUNC()(cx.b, y, alpha);

    return cr.storage;
  }
  
  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const float c1, const float c2) {
    converter cx, cy, cz, cr;
    cx.storage = x;
    cy.storage = y;
    cz.storage = z;

    cr.a = FUNC()(cx.a, cy.a, cz.a, c1, c2);
    cr.b = FUNC()(cx.b, cy.b, cz.b, c1, c2);

    return cr.storage;
  }

  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{
  converter cw;
cw.storage = w;
converter cS3;
cS3.storage = S3;
converter cS4;
cS4.storage = S4;
converter cS5;
cS5.a = FUNC()(cw.a, cS3.a, cS4.a);
cS5.b = FUNC()(cw.b, cS3.b, cS4.b);
return cS5.storage;
}
__device__ PackType LAMBWeightUpdate(const PackType w, float ratio, const PackType rLambdaWeight) const {
  converter cw;
  cw.storage = w;
  converter cS3;
  cS3.storage = rLambdaWeight;
  converter cS5;
  cS5.a = FUNC()(cw.a, ratio, cS3.a);
  cS5.b = FUNC()(cw.b, ratio, cS3.b);
  return cS5.storage;
}

  __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal,  curandState* randState, float val) const {
  }
};

template<class FUNC>
struct MULTI<FUNC, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y));
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const double alpha) {
    double rv = FUNC()(__longlong_as_double(x), alpha);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const double alpha) {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y), alpha);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const double y, const int alpha) {
    double rv = FUNC()(__longlong_as_double(x), y, alpha);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const double c1, const double c2) {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y), __longlong_as_double(z), c1, c2);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, double ratio, const PackType rLambdaWeight) const {}
    __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal,  curandState* randState, float val) const {
    }
};

template<class FUNC>
struct MULTI<FUNC, uint64_t> {
  static_assert(sizeof(PackType) == sizeof(uint64_t),
      "PackType must be the same size as uint64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    uint64_t rv = FUNC()(x, y);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const uint64_t alpha) {
    uint64_t rv = FUNC()((uint64_t)x, alpha);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const uint64_t alpha) {
    uint64_t rv = FUNC()((uint64_t)x, (uint64_t)y, alpha);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const uint64_t y, const int alpha) {
    uint64_t rv = FUNC()((uint64_t)x, (uint64_t)y, alpha);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const uint64_t c1, const uint64_t c2) {
    return FUNC()((uint64_t)x, (uint64_t)y, (uint64_t)z, c1, c2);
    return x;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, uint64_t ratio, const PackType rLambdaWeight) const {}
    __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const {
    }
};

template<class FUNC>
struct MULTI<FUNC, int64_t> {
  static_assert(sizeof(PackType) == sizeof(int64_t),
      "PackType must be the same size as int64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    int64_t rv = FUNC()((int64_t)x, (int64_t)y);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const int64_t alpha) {
    return FUNC()((int64_t)x, alpha);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const int64_t alpha) {
    return FUNC()((int64_t)x, (int64_t)y, alpha);
  }

  __device__ PackType operator()(const PackType x, const int64_t y, const int alpha) {
    return FUNC()((int64_t)x, (int64_t)y, alpha);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const int64_t c1, const int64_t c2) {
    return FUNC()((int64_t)x, (int64_t)y, (int64_t)z, c1, c2);
  }
  
  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, int64_t ratio, const PackType rLambdaWeight) const {}
    __device__ PackType dropout(const PackType x,  const PackType addTensorVal, const PackType biasVal, curandState* randState, float val) const {
    }
};

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

#if CUDART_VERSION < 9000
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r.x = ptr->x;
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ptr->x = val.x;
}
#else
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r = ((half*)ptr)[0];
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ((half*)ptr)[0] = val;
}
#endif

typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(Pack128& x, Pack128& y) {
    x.x = MULTI<FUNC, T>()(x.x, y.x);
    x.y = MULTI<FUNC, T>()(x.y, y.y);
  }

  __device__ void operator()(Pack128& x, T alpha) {
    x.x = MULTI<FUNC, T>()(x.x, alpha);
    x.y = MULTI<FUNC, T>()(x.y, alpha);
  }

  __device__ void operator()(Pack128& x, Pack128& y, T alpha) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, alpha);
    x.y = MULTI<FUNC, T>()(x.y, y.y, alpha);
  }

  __device__ void operator()(Pack128& x, Pack128& y, Pack128& z, T alpha) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, z.x, alpha);
    x.y = MULTI<FUNC, T>()(x.y, y.y, z.y, alpha);
  }

  __device__ void operator()(Pack128& x, T beta, int alpha) {
    x.x = MULTI<FUNC, T>()(x.x, beta, alpha);
    x.y = MULTI<FUNC, T>()(x.y, beta, alpha);
  }

   __device__ void operator()(Pack128& x, Pack128& y, T z, int alpha) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, z, alpha);
    x.y = MULTI<FUNC, T>()(x.y, y.y, z, alpha);
  }

   __device__ void operator()(Pack128& x, Pack128& y, Pack128& z, T alpha, T epsilon) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, z.x, alpha, epsilon);
    x.y = MULTI<FUNC, T>()(x.y, y.y, z.y, alpha, epsilon);
  }

  __device__ void r(Pack128& w, Pack128& S3, Pack128& S4, Pack128& S5) {
    S5.x = MULTI<FUNC, T>().r(w.x, S3.x, S4.x);
    S5.y = MULTI<FUNC, T>().r(w.y, S3.y, S4.y);
  }

  __device__ void LAMBWeightUpdate(Pack128& w, T ratio, Pack128& rLambdaWeight, Pack128& S5) {
    S5.x = MULTI<FUNC, T>().LAMBWeightUpdate(w.x, ratio, rLambdaWeight.x);
    S5.y = MULTI<FUNC, T>().LAMBWeightUpdate(w.y, ratio, rLambdaWeight.y);
  }

  __device__ PackType dropout(Pack128& w,  Pack128& addTensorVal, Pack128& biasVal, curandState* randState, float val, Pack128& S5) const {
    S5.x = MULTI<FUNC, T>().dropout(w.x, addTensorVal.x, biasVal.x, randState, val);
    S5.y = MULTI<FUNC, T>().dropout(w.y, addTensorVal.y, biasVal.y, randState, val);
  }
};

// #define LOAD_USING_64BITS
// #define STORE_USING_64BITS

typedef uint64_t LDType;

inline __device__ void Fetch128SRC(Pack128& v, const Pack128* p) {
#ifdef LOAD_USING_64BITS
  //TODO: this is incomplete. use shuffle to transfer v.x and v.y to correct threads

  LDType* ptr = (LDType*)(((uint8_t*)p) - (sizeof(Pack128) - sizeof(LDType)) * (threadIdx.x % WARP_SIZE));

  // uint64_t val = __shfl_sync(0xFFFFFFFF, v.x, , 32);
  asm volatile("ld.volatile.global.u64 {%0}, [%1];" : "=l"(v.x) : "l"((LDType*)ptr) : "memory");
  ptr = (LDType*)((uint8_t*)ptr + sizeof(Pack128)/((sizeof(Pack128)/sizeof(LDType))) * WARP_SIZE);
  asm volatile("ld.volatile.global.u64 {%0}, [%1];" : "=l"(v.y) : "l"((LDType*)ptr) : "memory");
#else
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
#endif
}

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
#ifdef LOAD_USING_64BITS
  //TODO: this is incomplete. use shuffle to transfer v.x and v.y to correct threads
  LDType* ptr = (LDType*)(((uint8_t*)p) - (sizeof(Pack128) - sizeof(LDType)) * (threadIdx.x % WARP_SIZE));
  asm volatile("ld.volatile.global.u64 {%0}, [%1];" : "=l"(v.x) : "l"((LDType*)ptr) : "memory");
  ptr = (LDType*)((uint8_t*)ptr + sizeof(Pack128)/((sizeof(Pack128)/sizeof(LDType))) * WARP_SIZE);
  asm volatile("ld.volatile.global.u64 {%0}, [%1];" : "=l"(v.y) : "l"((LDType*)ptr) : "memory");
#else
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
#endif
}
inline __device__ void Fetch128L2Cache(Pack128& v, const Pack128* p) {
  //asm volatile("ld.global.cg.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}
inline __device__ void Store128DST(Pack128* p, Pack128& v) {
#ifdef STORE_USING_64BITS
  //TODO: this is incomplete. use shuffle to transfer v.x and v.y to correct threads
  LDType* ptr = (LDType*)(((uint8_t*)p) - (sizeof(Pack128) - sizeof(LDType)) * (threadIdx.x % WARP_SIZE));
  asm volatile("st.volatile.global.u64 [%0], {%1};" :: "l"((LDType*)ptr), "l"(v.x): "memory");
  ptr = (LDType*)((uint8_t*)ptr + sizeof(Pack128)/((sizeof(Pack128)/sizeof(LDType))) * WARP_SIZE);
  asm volatile("st.volatile.global.u64 [%0], {%1};" :: "l"((LDType*)ptr), "l"(v.y): "memory");
#else
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
#endif
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
  #ifdef STORE_USING_64BITS
    //TODO: this is incomplete. use shuffle to transfer v.x and v.y to correct threads
    LDType* ptr = (LDType*)(((uint8_t*)p) - (sizeof(Pack128) - sizeof(LDType)) * (threadIdx.x % WARP_SIZE));
    asm volatile("st.volatile.global.u64 [%0], {%1};" :: "l"((LDType*)ptr), "l"(v.x): "memory");
    ptr = (LDType*)((uint8_t*)ptr + sizeof(Pack128)/((sizeof(Pack128)/sizeof(LDType))) * WARP_SIZE);
    asm volatile("st.volatile.global.u64 [%0], {%1};" :: "l"((LDType*)ptr), "l"(v.y): "memory");
  #else
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
  #endif
}

// template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE>
// __device__ __forceinline__ void ReduceCopyMulti(const int tid, const int nthreads,
//     int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
//     const int offset, const int N) {
//   for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
//     T val = vFetch(srcs[0]+idx);
//     #pragma unroll
//     for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
//     #pragma unroll 1
//     for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
//     if (WEIGHT_UPDATE) {
//       assert(false);
//       #pragma unroll
//       for (int i=0; i<MINDSTS; i++) {
//         T update = FuncSub2()(vFetch(((float*)dsts[i])+idx), (float)val);
//         vStore(dsts[i]+idx, update);
//       } 
//       #pragma unroll 1
//       for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) 
//       {
//         T update = FuncSub2()(vFetch(((float*)dsts[i])+idx), (float)val);
//         vStore(dsts[i]+idx, update);
//       }
//     } else {
//       assert(false);
//       #pragma unroll
//       for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
//       #pragma unroll 1
//       for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
//     }
//   }
// }

template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE>
__device__ __forceinline__ void ReduceCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    const int offset, const int N, T alpha, T beta1, T beta2, const int epoch, T* m, T* v,
    T* rStorage, const size_t mvStartOffset, int partStartOffset, int partSize, double* weightNorm, double* rNorm, const size_t buffNumElements) {

  double perThreadWeightNorm = 0, perThreadRNorm = 0;
  for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
    T val = vFetch(srcs[0]+idx);
    #pragma unroll
    for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));

    if (WEIGHT_UPDATE) {
      const size_t totalOffset = (mvStartOffset + idx);//%(totalSize/nranks);
      const size_t mOffset = partStartOffset + totalOffset%partSize;
      // size_t mOffset = idx;
      T m_ = vFetch(m + mOffset);
      T v_ = vFetch(v + mOffset);
      T wght_ = vFetch(dsts[0]+idx);
      
      m_ = FuncFirstMomentUpdate<T>()(m_, val, beta1);
      vStore(m + mOffset, m_);

      v_ = FuncSecondMomentUpdate<T>()(v_, val, beta2);
      vStore(v + mOffset, v_);

      m_ = FuncBiasCorrection<T>()(m_, beta1, epoch+1);
      v_ = FuncBiasCorrection<T>()(v_, beta2, epoch+1);

      if (LAMB) {
        perThreadWeightNorm += ((double)(wght_*wght_))/buffNumElements;
        // perThreadWeightNorm += ((double)(wght_*wght_))/1;
        T r_ = r<T>()(wght_, m_, v_);
        perThreadRNorm += ((double)(r_*r_))/buffNumElements;
        vStore(rStorage + mOffset, r_);
      } else {
        val = FuncAdamWeightUpdate<T>()(wght_, m_, v_, alpha, 1e-6);      
      }
    } else if (LAMB_SEND_COMPUTE) {
      const size_t totalOffset = (mvStartOffset + idx);//%(totalSize/nranks);
      const size_t mOffset = partStartOffset + totalOffset%partSize;
      
      double scale = ((*weightNorm > 0) ? (*rNorm > 0 ? *weightNorm/(*rNorm) : 1.0f) : 1.0f)/(*rNorm);
      val = LAMBWeightUpdate<T>()(val, (T)alpha*(T)scale, *(rStorage + mOffset));
      vStore((T*)srcs[0]+idx, val);
    }

    #pragma unroll
    for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
  }

  if (LAMB and WEIGHT_UPDATE) {
    atomicAdd(weightNorm, perThreadWeightNorm);
    atomicAdd(rNorm, perThreadRNorm);
  }
}

union LasF {
  float f1, f2;
  uint64_t l;
};

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE, int DROPOUT_BIAS_LAYERNORM>
__device__ __forceinline__ void ReduceCopy128bMulti( const int w, const int nw, const int t,
    int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
    T* firstMoment, T* secondMoment, T* rStorage,
    const int elemOffset, const int Npack, const T alpha, const T beta1, const T beta2, const int epoch,
    const size_t mvStartOffset, int partStartOffset, int partSize, double* weightNorm, double* rNorm, const size_t buffNumElements,
    curandState* randState) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;
  //Pack128* firstMomentPacked = ((Pack128*)(firstMoment+elemOffset))+offset;
  //Pack128* secondMomentPacked = ((Pack128*)(secondMoment+elemOffset))+offset;
  double perThreadWeightNorm = 0.0f;
  double perThreadRNorm = 0.0f;
  // if (LAMB_SEND_COMPUTE) {
  //   if (threadIdx.x == 0) {
  //           printf("rStorage %p\n", rStorage);
  //         }
  // }
  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);

    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }

    // Store
    if (WEIGHT_UPDATE) {
      if (firstMoment != NULL and secondMoment != NULL) {
        //ADAM
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 wght, m, v;
          Pack128 _vals = vals[u];
          Fetch128(wght, dsts[0]+u*WARP_SIZE);

          const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));//%(totalSize/nranks);
          const size_t mOffset = partStartOffset + totalOffset%partSize;

          Pack128* firstMomentPacked = (Pack128*)(firstMoment + mOffset);
          Pack128* secondMomentPacked = (Pack128*)(secondMoment + mOffset);
          Pack128* rStoragePack = (Pack128*)(rStorage + mOffset);

          Fetch128(m, firstMomentPacked);
          Fetch128(v, secondMomentPacked);
          // float4 mf4 = *(reinterpret_cast<float4*>(&m));
          // if (mf4.x != 0.0) {
          //     printf("844: mf4.x %f totalOffset %ld mvStartOffset %ld threadIdx.x %d secondMoment %p firstMoment %p buffNumElements %ld\n", mf4.x, totalOffset, mvStartOffset, threadIdx.x, secondMoment, firstMoment, buffNumElements);
          // }
          // if (buffNumElements == 2048 and totalOffset < 31260672) {
          //   printf("845: totalOffset %ld\n", totalOffset);
          // }
          MULTI128<FuncFirstMomentUpdate<T>, T>()(m, _vals, beta1);
          Store128(firstMomentPacked, m);
          
          MULTI128<FuncSecondMomentUpdate<T>, T>()(v, _vals, beta2);
          Store128(secondMomentPacked, v);

          MULTI128<FuncBiasCorrection<T>, T>()(m, beta1, epoch+1);

          MULTI128<FuncBiasCorrection<T>, T>()(v, beta2, epoch+1);
          
          if (LAMB) {
            float4 f4 = *(reinterpret_cast<float4*>(&wght));
            perThreadWeightNorm += ((double)(f4.x * f4.x))/buffNumElements + ((double)(f4.y * f4.y))/buffNumElements + 
                                   ((double)(f4.z * f4.z))/buffNumElements + ((double)(f4.w * f4.w))/buffNumElements;
            // perThreadWeightNorm += ((double)(f4.x * f4.x))/1 + ((double)(f4.y * f4.y))/1 + 
            //                        ((double)(f4.z * f4.z))/1 + ((double)(f4.w * f4.w))/1;
            Pack128 r_;
            MULTI128<r<T>, T>().r(wght, m, v, r_);
            f4 = *(reinterpret_cast<float4*>(&r_));
            
            perThreadRNorm += ((double)(f4.x * f4.x))/buffNumElements + ((double)(f4.y * f4.y))/buffNumElements + 
                              ((double)(f4.z * f4.z))/buffNumElements + ((double)(f4.w * f4.w))/buffNumElements;
            Store128(rStoragePack, r_);
            // float4 rf4 = *(reinterpret_cast<float4*>(&r_));
            // float4 mf4 = *(reinterpret_cast<float4*>(&m));
            // // if (fabs(rf4.x - 2.0)/2.0 > 1e-5) {
            // //     printf("862: rf4.x %f mf4.x %f\n", rf4.x, mf4.x);
            // // }
          } else {
            MULTI128<FuncAdamWeightUpdate<T>, T>()(wght, m, v, alpha, 1e-6);
          }

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, wght);
          }
        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, wght);
          }
        }
      } else {
        //SGD
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 val2;
          Fetch128(val2, dsts[0]+u*WARP_SIZE);
          Pack128 _vals = vals[u];
          MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, _vals);
          }

        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, _vals);
          }
        }
      }
    } else if (LAMB_SEND_COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));
          const size_t mOffset = partStartOffset + totalOffset%partSize;
          Pack128 rLambdaW;
          Fetch128(rLambdaW, (Pack128*)(rStorage+mOffset));
          double scale = ((*weightNorm > 0) ? (*rNorm > 0 ? *weightNorm/(*rNorm) : 1.0f) : 1.0f)/(*rNorm);
          Pack128 finalVal;
          MULTI128<LAMBWeightUpdate<T>, T>().LAMBWeightUpdate(vals[u], alpha*(T)scale, rLambdaW, finalVal);
          float4 f4 = *(reinterpret_cast<float4*>(&finalVal));
          // float4 rf4 = *(reinterpret_cast<float4*>(&rLambdaW));
          // if (buffNumElements == 31260672 && fabs(f4.x - 0.5)/0.5 > 1e-5) {
          //     printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.x %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.x);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.y - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.y %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.y);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.z - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.z %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.z);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.w - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.w %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.w);
          // }

          // if (buffNumElements == 2048) {
          //   f4.x = 4.f/3.f;
          //   f4.y = 4.f/3.f;
          //   f4.z = 4.f/3.f;
          //   f4.w = 4.f/3.f;
          //   finalVal = *(reinterpret_cast<Pack128*>(&f4));
          // } else {
          //   f4.x = 0.5f;
          //   f4.y = 0.5f;
          //   f4.z = 0.5f;
          //   f4.w = 0.5f;
          //   finalVal = *(reinterpret_cast<Pack128*>(&f4));
          // }
          // if (buffNumElements == 31260672 && totalOffset >= 31260672) {
          //   printf("f4.x %f totalOffset %ld\n", f4.x, totalOffset);
          // }
          Store128((Pack128*)srcs[0]+u*WARP_SIZE, finalVal);

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, finalVal);
          }
        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, finalVal);
          }
        }
    } else if (DROPOUT_BIAS_LAYERNORM) {
      for (int u = 0; u < UNROLL; ++u) {
        Pack128 _vals = vals[u];
        //firstMoment is addTensor, secondMoment is bias, and epoch is biasSize when DROPOUT_BIAS_LAYERNORM is enabled
        const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));
        const size_t biasOffset = totalOffset%epoch;
        Pack128 addTensorVal;
        Fetch128(addTensorVal, (Pack128*)(firstMoment+totalOffset));
        Pack128 biasVal;
        Fetch128(biasVal, (Pack128*)(secondMoment+biasOffset));
        MULTI128<FuncDropout<T>, T>().dropout(_vals, addTensorVal, biasVal, randState, 0.1f, _vals);

        for (int i = 0; i < MINDSTS; i++) {
          Store128(dsts[i]+u*WARP_SIZE, _vals);
        }

      #pragma unroll 1
        for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          Store128(dsts[i]+u*WARP_SIZE, _vals);
        }
      }
    } else {
      for (int i = 0; i < MINDSTS; i++) {
        for (int u = 0; u < UNROLL; ++u) {
          Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }
      #pragma unroll 1
      for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        for (int u = 0; u < UNROLL; ++u) {
          Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    // firstMomentPacked += inc;
    // secondMomentPacked += inc;
    offset += inc;
  }

  if (LAMB and WEIGHT_UPDATE) {
    atomicAdd(weightNorm, perThreadWeightNorm);
    atomicAdd(rNorm, perThreadRNorm);
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

// Try to limit consecutive load/stores to 8.
// Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
#define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE, int DROPOUT_BIAS_LAYERNORM>
__device__ __forceinline__ void ReduceOrCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    T* firstMoment, T* secondMoment, T* rStorage,
    int N, const T alpha, const T beta1, const T beta2, const int epoch, const size_t mvStartOffset, int partStartOffset, 
    int partSize, double* weightNorm, double* rNorm, const size_t buffNumElements, curandState* randState) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int alignDiff = 0;
  int align = ptrAlign128(srcs[0]);
  #pragma unroll
  for (int i=1; i<MINSRCS; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  if (firstMoment)
    alignDiff |= (align ^ ptrAlign128(firstMoment));
  if (secondMoment)
    alignDiff |= (align ^ ptrAlign128(secondMoment));  
  if (rStorage)
    alignDiff |= (align ^ ptrAlign128(rStorage));

  int Npreamble = alignDiff ? Nrem :
    N < alignof(Pack128) ? N :
    (alignof(Pack128) - align) % alignof(Pack128);

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment

  if (Npreamble) {
    ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>
    (tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble, alpha, beta1, beta2, epoch, firstMoment, secondMoment, rStorage, mvStartOffset, 
    partStartOffset, partSize, weightNorm, rNorm, buffNumElements);
    Nrem -= Npreamble;
    if (Nrem == 0) return;
  }
  int offset = Npreamble;

  // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 128-bit alignable.
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  const int packFactor = sizeof(Pack128) / sizeof(T);
  // stage 2a: main loop
  int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
      * (AUTOUNROLL * WARP_SIZE); // round down
  int Nelem2a = Npack2a * packFactor;
 
  ReduceCopy128bMulti<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE, DROPOUT_BIAS_LAYERNORM>(w, nw, t, nsrcs, srcs, ndsts, dsts, firstMoment, secondMoment, rStorage, offset, Npack2a, alpha, beta1, beta2, epoch, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements, randState);

  Nrem -= Nelem2a;
  if (Nrem == 0) return;
  offset += Nelem2a;

  // stage 2b: slightly less optimized for section when we don't have full
  // unrolling

  int Npack2b = Nrem / packFactor;
  int Nelem2b = Npack2b * packFactor;

  ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE, DROPOUT_BIAS_LAYERNORM>(w, nw, t, nsrcs, srcs, ndsts, dsts, firstMoment, secondMoment, rStorage, offset, Npack2b, alpha, beta1, beta2, epoch, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements, randState);

  Nrem -= Nelem2b;
  if (Nrem == 0) return;
  offset += Nelem2b;

  // stage 2c: tail
  ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem, alpha, beta1, beta2, epoch, firstMoment, secondMoment, rStorage, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements);
}

template<class FUNC, typename T>
__device__ 
void printHalfInPack128(int line, Pack128& packVal) {
  half* h = (half*)&packVal;
  printf("%d: h1 %f, h2 %f, h3 %f, h4 %f, h5 %f, h6 %f, h7 %f, h8 %f\n", 
         line, __half2float(h[0]), __half2float(h[1]), __half2float(h[2]), __half2float(h[3]), __half2float(h[4]),
         __half2float(h[5]), __half2float(h[6]), __half2float(h[7]));
}

template<class FUNC, typename T, int UNROLL, int SRC, int DST, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE, int DROPOUT_BIAS_LAYERNORM>
__device__ __forceinline__ void ReduceCopy128bMultiMatrixBlock(const int w, const int nw, const int t,
    int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
    T* firstMoment, T* secondMoment, T* rStorage,
    const int elemOffset, const int Npack, const T alpha, const T beta1, const T beta2, const int epoch,
    const size_t linearStartOffset, int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixRows, int matrixCols, 
    double* weightNorm, double* rNorm, const size_t buffNumElements, curandState* randState) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const size_t mvStartOffset = linearStartOffset;
  const int partStartOffset = 1;
  const int partSize = 1;
  
  const Pack128* srcs[MAXSRCS];
  if (SRC) {
    srcs[0] = (const Pack128*)(s[0]);
  }
  for (int i=SRC; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  if (DST) {
    dsts[0] = (Pack128*)(d[0]);
  }
  for (int i=DST; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;
  //Pack128* firstMomentPacked = ((Pack128*)(firstMoment+elemOffset))+offset;
  //Pack128* secondMomentPacked = ((Pack128*)(secondMoment+elemOffset))+offset;
  double perThreadWeightNorm = 0.0f;
  double perThreadRNorm = 0.0f;
  // if (LAMB_SEND_COMPUTE) {
  //   if (threadIdx.x == 0) {
  //           printf("rStorage %p\n", rStorage);
  //         }
  // }
  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    if (SRC) {
      //Load the elements of block: 'chunkRows' x 'chunkCols' starting at row 'chunkStartRow'.
      //Condition that chunkCols is a multiple of (sizeof(Pack128)/sizeof(T) .
      // assert(chunkCols % (sizeof(Pack128)/sizeof(T)) == 0);
      if (false && chunkStartRow == -1) {
        for (int u = 0; u < UNROLL; ++u) {
          const size_t totalOffset = (linearStartOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));
        
          Fetch128(vals[u], (const Pack128*)(s[0]+totalOffset));
        }
      } else {
        for (int u = 0; u < UNROLL; ++u) {
          const size_t chunkElemOffset = (linearStartOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));

          const int chunkElemRow = chunkElemOffset / chunkCols;
          const int chunkElemCol = chunkElemOffset % chunkCols;
          
          Fetch128SRC(vals[u], (const Pack128*)(s[0]+((chunkStartRow + chunkElemRow) * matrixCols + (chunkStartCol + chunkElemCol))));
        }
      }
    } else {
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);
    }

    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }

    // Store
    if (false && WEIGHT_UPDATE) {
      if (firstMoment != NULL and secondMoment != NULL) {
        //ADAM
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 wght, m, v;
          Pack128 _vals = vals[u];
          Fetch128(wght, dsts[0]+u*WARP_SIZE);

          const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));//%(totalSize/nranks);
          const size_t mOffset = partStartOffset + totalOffset%partSize;

          Pack128* firstMomentPacked = (Pack128*)(firstMoment + mOffset);
          Pack128* secondMomentPacked = (Pack128*)(secondMoment + mOffset);
          Pack128* rStoragePack = (Pack128*)(rStorage + mOffset);

          Fetch128(m, firstMomentPacked);
          Fetch128(v, secondMomentPacked);
          // float4 mf4 = *(reinterpret_cast<float4*>(&m));
          // if (mf4.x != 0.0) {
          //     printf("844: mf4.x %f totalOffset %ld mvStartOffset %ld threadIdx.x %d secondMoment %p firstMoment %p buffNumElements %ld\n", mf4.x, totalOffset, mvStartOffset, threadIdx.x, secondMoment, firstMoment, buffNumElements);
          // }
          // if (buffNumElements == 2048 and totalOffset < 31260672) {
          //   printf("845: totalOffset %ld\n", totalOffset);
          // }
          MULTI128<FuncFirstMomentUpdate<T>, T>()(m, _vals, beta1);
          Store128(firstMomentPacked, m);
          
          MULTI128<FuncSecondMomentUpdate<T>, T>()(v, _vals, beta2);
          Store128(secondMomentPacked, v);

          MULTI128<FuncBiasCorrection<T>, T>()(m, beta1, epoch+1);

          MULTI128<FuncBiasCorrection<T>, T>()(v, beta2, epoch+1);
          
          if (LAMB) {
            float4 f4 = *(reinterpret_cast<float4*>(&wght));
            perThreadWeightNorm += ((double)(f4.x * f4.x))/buffNumElements + ((double)(f4.y * f4.y))/buffNumElements + 
                                   ((double)(f4.z * f4.z))/buffNumElements + ((double)(f4.w * f4.w))/buffNumElements;
            // perThreadWeightNorm += ((double)(f4.x * f4.x))/1 + ((double)(f4.y * f4.y))/1 + 
            //                        ((double)(f4.z * f4.z))/1 + ((double)(f4.w * f4.w))/1;
            Pack128 r_;
            MULTI128<r<T>, T>().r(wght, m, v, r_);
            f4 = *(reinterpret_cast<float4*>(&r_));
            
            perThreadRNorm += ((double)(f4.x * f4.x))/buffNumElements + ((double)(f4.y * f4.y))/buffNumElements + 
                              ((double)(f4.z * f4.z))/buffNumElements + ((double)(f4.w * f4.w))/buffNumElements;
            Store128(rStoragePack, r_);
            // float4 rf4 = *(reinterpret_cast<float4*>(&r_));
            // float4 mf4 = *(reinterpret_cast<float4*>(&m));
            // // if (fabs(rf4.x - 2.0)/2.0 > 1e-5) {
            // //     printf("862: rf4.x %f mf4.x %f\n", rf4.x, mf4.x);
            // // }
          } else {
            MULTI128<FuncAdamWeightUpdate<T>, T>()(wght, m, v, alpha, 1e-6);
          }

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, wght);
          }
        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, wght);
          }
        }
      } else {
        //SGD
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 val2;
          Fetch128(val2, dsts[0]+u*WARP_SIZE);
          Pack128 _vals = vals[u];
          MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, _vals);
          }

        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, _vals);
          }
        }
      }
    } else if (false && LAMB_SEND_COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));
          const size_t mOffset = partStartOffset + totalOffset%partSize;
          Pack128 rLambdaW;
          Fetch128(rLambdaW, (Pack128*)(rStorage+mOffset));
          double scale = ((*weightNorm > 0) ? (*rNorm > 0 ? *weightNorm/(*rNorm) : 1.0f) : 1.0f)/(*rNorm);
          Pack128 finalVal;
          MULTI128<LAMBWeightUpdate<T>, T>().LAMBWeightUpdate(vals[u], alpha*(T)scale, rLambdaW, finalVal);
          float4 f4 = *(reinterpret_cast<float4*>(&finalVal));
          // float4 rf4 = *(reinterpret_cast<float4*>(&rLambdaW));
          // if (buffNumElements == 31260672 && fabs(f4.x - 0.5)/0.5 > 1e-5) {
          //     printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.x %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.x);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.y - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.y %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.y);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.z - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.z %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.z);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.w - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.w %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.w);
          // }

          // if (buffNumElements == 2048) {
          //   f4.x = 4.f/3.f;
          //   f4.y = 4.f/3.f;
          //   f4.z = 4.f/3.f;
          //   f4.w = 4.f/3.f;
          //   finalVal = *(reinterpret_cast<Pack128*>(&f4));
          // } else {
          //   f4.x = 0.5f;
          //   f4.y = 0.5f;
          //   f4.z = 0.5f;
          //   f4.w = 0.5f;
          //   finalVal = *(reinterpret_cast<Pack128*>(&f4));
          // }
          // if (buffNumElements == 31260672 && totalOffset >= 31260672) {
          //   printf("f4.x %f totalOffset %ld\n", f4.x, totalOffset);
          // }
          Store128((Pack128*)srcs[0]+u*WARP_SIZE, finalVal);

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, finalVal);
          }
        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, finalVal);
          }
        }
    } else if (false && DROPOUT_BIAS_LAYERNORM) {
      for (int u = 0; u < UNROLL; ++u) {
        Pack128 _vals = vals[u];
        //firstMoment is addTensor, secondMoment is bias, and epoch is biasSize when DROPOUT_BIAS_LAYERNORM is enabled
        const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));
        const size_t biasOffset = totalOffset%epoch;
        Pack128 addTensorVal;
        Fetch128(addTensorVal, (Pack128*)(firstMoment+totalOffset));
        Pack128 biasVal;
        Fetch128(biasVal, (Pack128*)(secondMoment+biasOffset));
        MULTI128<FuncDropout<T>, T>().dropout(_vals, addTensorVal, biasVal, randState, 0.1f, _vals);

        for (int i = 0; i < MINDSTS; i++) {
          Store128(dsts[i]+u*WARP_SIZE, _vals);
        }

      #pragma unroll 1
        for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          Store128(dsts[i]+u*WARP_SIZE, _vals);
        }
      }
    } else {
      if (DST) {
        if (false && chunkStartRow == -1) {
          for (int u = 0; u < UNROLL; ++u) {
            const size_t totalOffset = (linearStartOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));
            Store128((Pack128*)(d[0]+totalOffset), vals[u]);
            if (totalOffset == 0) {
              //printHalfInPack128<FUNC, T>(__LINE__, vals[u]);
            }
          }
        } else {
          for (int u = 0; u < UNROLL; ++u) {
            const size_t chunkElemOffset = (linearStartOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));

            const int chunkElemRow = chunkElemOffset / chunkCols;
            const int chunkElemCol = chunkElemOffset % chunkCols;
            
            
            Store128DST((Pack128*)(d[0]+((chunkStartRow + chunkElemRow) * matrixCols + (chunkStartCol + chunkElemCol))), vals[u]);
          }
        }
      }

      for (int i = DST; i < MINDSTS; i++) {
        for (int u = 0; u < UNROLL; ++u) {
          Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }
      #pragma unroll 1
      for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        for (int u = 0; u < UNROLL; ++u) {
          Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }
    }
    for (int i=SRC; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=DST; i<MAXDSTS; i++) dsts[i] += inc;
    // firstMomentPacked += inc;
    // secondMomentPacked += inc;
    offset += inc;
  }

  if (LAMB and WEIGHT_UPDATE) {
    atomicAdd(weightNorm, perThreadWeightNorm);
    atomicAdd(rNorm, perThreadRNorm);
  }
}

template<int UNROLL, class FUNC, typename T, int SRC, int DST, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE, int DROPOUT_BIAS_LAYERNORM>
__device__ __forceinline__ void ReduceOrCopyMultiMatrixBlock(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    T* firstMoment, T* secondMoment, T* rStorage,
    int N, const T alpha, const T beta1, const T beta2, const int epoch, const size_t linearStartOffset, 
    int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixRows, int matrixCols,
    double* weightNorm, double* rNorm, const size_t buffNumElements, curandState* randState) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int alignDiff = 0;
  int align = ptrAlign128(srcs[0]);
  #pragma unroll
  for (int i=1; i<MINSRCS; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  if (firstMoment)
    alignDiff |= (align ^ ptrAlign128(firstMoment));
  if (secondMoment)
    alignDiff |= (align ^ ptrAlign128(secondMoment));  
  if (rStorage)
    alignDiff |= (align ^ ptrAlign128(rStorage));

  int Npreamble = alignDiff ? Nrem :
    N < alignof(Pack128) ? N :
    (alignof(Pack128) - align) % alignof(Pack128);

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment

  if (Npreamble) {
    // ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>
    // (tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble, alpha, beta1, beta2, epoch, firstMoment, secondMoment, rStorage, mvStartOffset, 
    // partStartOffset, partSize, weightNorm, rNorm, buffNumElements);
    Nrem -= Npreamble;
    if (Nrem == 0) return;
  }
  int offset = Npreamble;

  // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 128-bit alignable.
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)
  
  const int packFactor = sizeof(Pack128) / sizeof(T);
  // stage 2a: main loop
  int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
      * (AUTOUNROLL * WARP_SIZE); // round down
  int Nelem2a = Npack2a * packFactor;
 
  ReduceCopy128bMultiMatrixBlock<FUNC, T, AUTOUNROLL, SRC, DST, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE, DROPOUT_BIAS_LAYERNORM>(w, nw, t, nsrcs, srcs, ndsts, dsts, firstMoment, secondMoment, rStorage, offset, Npack2a, alpha, beta1, beta2, epoch, linearStartOffset + offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixRows, matrixCols, weightNorm, rNorm, buffNumElements, randState);

  Nrem -= Nelem2a;
  if (Nrem == 0) return;
  offset += Nelem2a;

  // stage 2b: slightly less optimized for section when we don't have full
  // unrolling

  int Npack2b = Nrem / packFactor;
  int Nelem2b = Npack2b * packFactor;

  ReduceCopy128bMultiMatrixBlock<FUNC, T, 1, SRC, DST, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE, DROPOUT_BIAS_LAYERNORM>(w, nw, t, nsrcs, srcs, ndsts, dsts, firstMoment, secondMoment, rStorage, offset, Npack2b, alpha, beta1, beta2, epoch, linearStartOffset + offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixRows, matrixCols, weightNorm, rNorm, buffNumElements, randState);

  Nrem -= Nelem2b;
  if (Nrem == 0) return;
  offset += Nelem2b;

  // stage 2c: tail
  // ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem, alpha, beta1, beta2, epoch, firstMoment, secondMoment, rStorage, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements);
}

#endif // COMMON_KERNEL_H_