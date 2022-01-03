/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include <cstdio>
#include <cstdint>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TYPE_COMMON_KERNEL 0









#if TYPE_COMMON_KERNEL == 0

  // Define min for ssize_t
  static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

  typedef uint64_t PackType;

  // unpack x and y to elements of type T and apply FUNC to each element
  template<class FUNC, typename T>
  struct MULTI {
    __device__ PackType operator()(const PackType x, const PackType y) const;

    __device__ T atomicAdd(T* ptr, const PackType y) const;
      __device__ uint64_t mixedbinOp1(const  T beta2, const uint64_t v, const uint32_t S0) const;
    __device__ uint64_t binOp2(const  T beta2, const uint64_t S2) const;
    __device__ uint64_t mixedbinOp3(const  T beta1, const uint64_t m, const uint32_t S0) const;
    __device__ uint64_t binOp4(const  T beta1, const uint64_t S1) const;
    __device__ uint64_t delta(const  T lr, const uint64_t S3, const uint64_t S4) const;
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const;

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

    __device__ int8_t atomicAdd(int8_t* ptr, const PackType y) {}
    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const int8_t beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const int8_t beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const int8_t beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const int8_t beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const int8_t lr, const uint64_t S3, const uint64_t S4) const {
  }
  __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) {}
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

    __device__ uint8_t atomicAdd(uint8_t* ptr, const PackType y) {}

    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const uint8_t beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const uint8_t beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const uint8_t beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const uint8_t beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const uint8_t lr, const uint64_t S3, const uint64_t S4) const {
  }
  __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) {}
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

    __device__ int32_t atomicAdd(int32_t* ptr, const PackType y) {
      converter cx;
      cx.storage = y;

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }
    
    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const int32_t beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const int32_t beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const int32_t beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const int32_t beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const int32_t lr, const uint64_t S3, const uint64_t S4) const {
  }
  __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) {}
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

    __device__ uint32_t atomicAdd(uint32_t* ptr, const PackType y) {
      converter cx;
      cx.storage = y;

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }

    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const uint32_t beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const uint32_t beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const uint32_t beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const uint32_t beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const uint32_t lr, const uint64_t S3, const uint64_t S4) const {
  }
  __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) {}
  };

  template<class FUNC>
  struct MULTI<FUNC, half> {
    static_assert(sizeof(PackType) == 4 * sizeof(half),
        "PackType must be four times the size of half.");

    struct PackHalf2 {
      half2 a, b;
    };

    __device__ PackType operator()(const PackType x, const PackType y) const {
      struct PackHalf2 cx, cy, cr;
      cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
      cy = *(reinterpret_cast<const struct PackHalf2*>(&y));

      cr.a = FUNC()(cx.a, cy.a);
      cr.b = FUNC()(cx.b, cy.b);

      return *(reinterpret_cast<PackType*>(&cr));
    }

      __device__ half atomicAdd(half* ptr, const PackType y) {
      // PackHalf2 cx;
      // cx = *(reinterpret_cast<const struct PackHalf2*>(&y));

      // ::atomicAdd(ptr, cx.a);
      // return ::atomicAdd(ptr, cx.b);
    }

    __device__ half2 atomicAdd(half2* ptr, const PackType y) {
      PackHalf2 cx;
      cx = *(reinterpret_cast<const struct PackHalf2*>(&y));

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }

    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const {
      converterfloat cw;
      cw.storage = w;
      converterfloat cdelta;
      cdelta.storage = delta;
      converterfloat cS2;
      cS2.FOO.x0 = FUNC()(cw.FOO.getx0(), cdelta.FOO.getx0());
      cS2.FOO.x1 = FUNC()(cw.FOO.getx1(), cdelta.FOO.getx1());
      //assert(cS2.FOO.x0 == 2.0f && cS2.FOO.x1 == 2.0f);
      return cS2.storage;
    }

    __device__ uint64_t mixedbinOp1(const float beta2, const uint64_t v, const uint32_t S0) const {
  converterfloat cv;
  cv.storage = v;
  converterhalf cS0;
  cS0 = *(reinterpret_cast<const converterhalf*>(&S0));
  converterfloat cS2;
  cS2.FOO.x0 = FUNC()(beta2, cv.FOO.getx0(), cS0.getx0());
  cS2.FOO.x1 = FUNC()(beta2, cv.FOO.getx1(), cS0.getx1());
  //assert(cS2.FOO.x0 == 2.0f && cS2.FOO.x1 == 2.0f);
  return cS2.storage;
  }
    __device__ uint64_t binOp2(const float beta2, const uint64_t S2) const {
  converterfloat cS2;
  cS2.storage = S2;
  converterfloat cS4;
  cS4.FOO.x0 = FUNC()(beta2, cS2.FOO.getx0());
  cS4.FOO.x1 = FUNC()(beta2, cS2.FOO.getx1());
  return cS4.storage;
  }
    __device__ uint64_t mixedbinOp3(const float beta1, const uint64_t m, const uint32_t S0) const {
  converterfloat cm;
  cm.storage = m;
  converterhalf cS0;
  cS0 = *(reinterpret_cast<const converterhalf*>(&S0));
  converterfloat cS1;
  cS1.FOO.x0 = FUNC()(beta1, cm.FOO.getx0(), cS0.getx0());
  cS1.FOO.x1 = FUNC()(beta1, cm.FOO.getx1(), cS0.getx1());
  return cS1.storage;
  }
    __device__ uint64_t binOp4(const float beta1, const uint64_t S1) const {
  converterfloat cS1;
  cS1.storage = S1;
  converterfloat cS3;
  cS3.FOO.x0 = FUNC()(beta1, cS1.FOO.getx0());
  cS3.FOO.x1 = FUNC()(beta1, cS1.FOO.getx1());
  return cS3.storage;
  }
    __device__ uint64_t delta(const float lr, const uint64_t S3, const uint64_t S4) const {
  converterfloat cS3;
  cS3.storage = S3;
  converterfloat cS4;
  cS4.storage = S4;
  converterfloat cS5;
  cS5.FOO.x0 = FUNC()(lr, cS3.FOO.getx0(), cS4.FOO.getx0());
  cS5.FOO.x1 = FUNC()(lr, cS3.FOO.getx1(), cS4.FOO.getx1());
  return cS5.storage;
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

    __device__ float atomicAdd(float* ptr, const PackType y) {
      converter cx;
      cx.storage = y;

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }

    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const float beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const float beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const float beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const float beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const float lr, const uint64_t S3, const uint64_t S4) const {
  }
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const {}
  };

  template<class FUNC>
  struct MULTI<FUNC, double> {
    static_assert(sizeof(PackType) == sizeof(double),
        "PackType must be the same size as double.");
    __device__ PackType operator()(const PackType x, const PackType y) const {
      double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y));
      return __double_as_longlong(rv);
    }

    __device__ double atomicAdd(double* ptr, const PackType y) {
      return ::atomicAdd(ptr, __longlong_as_double(y));
    }

    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const double beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const double beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const double beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const double beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const double lr,  const uint64_t S3, const uint64_t S4) const {
  }
  __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const {}
  };

  template<class FUNC>
  struct MULTI<FUNC, uint64_t> {
    static_assert(sizeof(PackType) == sizeof(uint64_t),
        "PackType must be the same size as uint64_t.");
    __device__ PackType operator()(const PackType x, const PackType y) const {
      uint64_t rv = FUNC()(x, y);
      return rv;
    }

    __device__ uint64_t atomicAdd(uint64_t* ptr, const PackType y) {
      return y;//::atomicAdd(ptr, y);
    }

    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const uint64_t beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const uint64_t beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const uint64_t beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const uint64_t beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const uint64_t lr, const uint64_t S3, const uint64_t S4) const {
  }
  __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const {}
  };

  template<class FUNC>
  struct MULTI<FUNC, int64_t> {
    static_assert(sizeof(PackType) == sizeof(int64_t),
        "PackType must be the same size as int64_t.");
    __device__ PackType operator()(const PackType x, const PackType y) const {
      int64_t rv = FUNC()((int64_t)x, (int64_t)y);
      return rv;
    }

    __device__ int64_t atomicAdd(int64_t* ptr, const PackType y) {
      return y;//::atomicAdd(ptr, (int64_t)y);
    }

    struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };

    __device__ uint64_t mixedbinOp1(const int64_t beta2, const uint64_t v, const uint32_t S0) const {
  }
    __device__ uint64_t binOp2(const int64_t beta2, const uint64_t S2) const {
  }
    __device__ uint64_t mixedbinOp3(const int64_t beta1, const uint64_t m, const uint32_t S0) const {
  }
    __device__ uint64_t binOp4(const int64_t beta1, const uint64_t S1) const {
  }
    __device__ uint64_t delta(const int64_t lr, const uint64_t S3, const uint64_t S4) const {
  }
  __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const {}
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

    __device__ void mixedbinOp1(const float beta2, Pack128& v, uint2 S0, Pack128& S2) {
      S2.x = MULTI<FUNC, T>().mixedbinOp1(beta2, v.x, S0.x);
      S2.y = MULTI<FUNC, T>().mixedbinOp1(beta2, v.y, S0.y);
    }
  __device__ void binOp2(const float beta2, Pack128& S2, Pack128& S4) {
      S4.x = MULTI<FUNC, T>().binOp2(beta2, S2.x);
      S4.y = MULTI<FUNC, T>().binOp2(beta2, S2.y);
    }
  __device__ void mixedbinOp3(const float beta1, Pack128& m, uint2 S0, Pack128& S1) {
      S1.x = MULTI<FUNC, T>().mixedbinOp3(beta1, m.x, S0.x);
      S1.y = MULTI<FUNC, T>().mixedbinOp3(beta1, m.y, S0.y);
    }
  __device__ void binOp4(const float beta1, Pack128& S1, Pack128& S3) {
      S3.x = MULTI<FUNC, T>().binOp4(beta1, S1.x);
      S3.y = MULTI<FUNC, T>().binOp4(beta1, S1.y);
    }
  __device__ void delta(const float lr, Pack128& S3, Pack128& S4, Pack128& S5) {
      S5.x = MULTI<FUNC, T>().delta(lr, S3.x, S4.x);
      S5.y = MULTI<FUNC, T>().delta(lr, S3.y, S4.y);
    }

    __device__ void weightUpdate(Pack128& w, Pack128& delta, Pack128& S5) {
      S5.x = MULTI<FUNC, T>().weightUpdate(w.x, delta.x);
      S5.y = MULTI<FUNC, T>().weightUpdate(w.y, delta.y);
    }
  };


  inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  }
  inline __device__ void Store128(Pack128* p, Pack128& v) {
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
  }

  struct Pack256 {
    Pack128 x, y;
  };

  inline __device__ void Fetch256(Pack256& v, const Pack256* p) {
    Fetch128(v.x, (const Pack128*)p);
    Fetch128(v.y, ((const Pack128*)p) + 1);
  }
  inline __device__ void Store256(Pack256* p, Pack256& v) {
    Store128((Pack128*)p, v.x);
    Store128(((Pack128*)p) + 1, v.y);
  }

  struct halfToUint64_t {
      half2 h1;
      half2 h2;
  };

  inline __device__ uint64_t float4ToHalf4(Pack128& v) {
    float2 h1 = *(reinterpret_cast<float2*>(&v.x));
    float2 h2 = *(reinterpret_cast<float2*>(&v.y));
    // assert (h1.x == -1.0f);
    // assert (h1.y == -1.0f);
    // assert (h1. == -1.0f);

    half2 r1 = __floats2half2_rn(h1.x, h1.y);
    half2 r2 = __floats2half2_rn(h2.x, h2.y);

    halfToUint64_t converter;
    converter.h1 = r1;
    converter.h2 = r2;

    return *(reinterpret_cast<uint64_t*>(&converter));
  }

  template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
  __device__ __forceinline__ void ReduceCopyMulti(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      const int offset, const int N) {
    for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
      T val = vFetch(srcs[0]+idx);
      #pragma unroll
      for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
      #pragma unroll 1
      for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));

      #pragma unroll
      for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
      #pragma unroll 1
      for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
    }
  }

  template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceCopy128bMultiComputation( const int w, const int nw, const int t,
      int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
      const int elemOffset, const int Npack, T lr, T beta1, T beta2, T* m, T* v) {
    const int inc = nw * UNROLL * WARP_SIZE;
    int offset = w * UNROLL * WARP_SIZE + t;

    const Pack128* srcs[MAXSRCS];
    for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
    Pack128* dsts[MAXDSTS];
    for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;
      Pack128* mPack = ((Pack128*)(m+elemOffset))+offset;
      Pack128* vPack = ((Pack128*)(v+elemOffset))+offset;

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


      if (false &&COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 readVal;
          Fetch128(readVal, dsts[0]+u*WARP_SIZE);
          Pack128 finalVal = vals[u];
          
          //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

          Store128(dsts[0]+u*WARP_SIZE, finalVal);

          for (int i = 1; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, finalVal);
          }

        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, finalVal);
          }
        }
      } else if (false && ALLGATHER_COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 readVal;
          Fetch128(readVal, dsts[0]+u*WARP_SIZE);
          Pack128 finalVal = vals[u];
          
          //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

          Store128(dsts[0]+u*WARP_SIZE, finalVal);

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, vals[u]);
          }

        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, vals[u]);
          }
        }
      } else {
        for (int u = 0; u < UNROLL; ++u) {
          // if (COMPUTE) {
          //   half* h1 = (half*)(&vals[u]);
          //   if (((float)h1[0]) != 4.0f) {
          //     Pack128 v1;
          //     Pack128 v2;
          //     Fetch128(v1, srcs[0]+u*WARP_SIZE);
          //     Fetch128(v2, srcs[1]+u*WARP_SIZE);

          //     half* hv1 = (half*)(&v1);
          //     half* hv2 = (half*)(&v2);

          //     printf("h1 %f hv1 %f hv2 %f AUTOUNROLL %d elemOffset %d offset %d Npack %d MINSRCS %d nsrcs %d\n", ((float)h1[0]), (float)hv1[0], (float)hv2[0], UNROLL, elemOffset, offset, Npack, MINSRCS, nsrcs);

          //   }
          // }
        }
          
        // Store
        for (int i = 0; i < MINDSTS; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
        #pragma unroll 1
        for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }

      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      offset += inc;
    }
  }

  struct FuncSumHalf {
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

  struct converterhalf1 {half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };

  union converterfloat4toulong2
  {
    Pack128 storage;
    struct { float x0, x1,x2,x3;};
  };

  typedef uint64_t gPackType;

  template<class FUNC, typename TF16, typename TF32, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceCopy128bMultiComputation2( const int w, const int nw, const int t,
      int nsrcs, const TF16* s[MAXSRCS], int ndsts, TF16* d[MAXDSTS],
      const int elemOffset, const int Npack, const int Nelem, TF32* weight, TF32 lr, TF32 beta1, TF32 beta2, TF32* m, TF32* v) {
    const int inc = nw * UNROLL * WARP_SIZE;
    int offset = w * UNROLL * WARP_SIZE + t;

    const gPackType* srcs[MAXSRCS];
    for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const gPackType*)(s[i]+elemOffset))+offset;
    gPackType* dsts[MAXDSTS];
    for (int i=0; i<MAXDSTS; i++) dsts[i] = ((gPackType*)(d[i]+elemOffset))+offset;
    Pack128* mPack = ((Pack128*)(m+elemOffset))+offset;
    Pack128* vPack = ((Pack128*)(v+elemOffset))+offset;
    Pack128* wPack = ((Pack128*)(weight+elemOffset))+offset;
    
    while (offset < Npack*2) {
      gPackType vals[UNROLL];
      // Load and reduce
      for (int u = 0; u < UNROLL; ++u) vals[u] = *(srcs[0]+u*WARP_SIZE);

      for (int i=1; i<MINSRCS; i++) {
        gPackType vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FuncSumHalf, TF16>()(vals[u], vals2[u]);
      }
      #pragma unroll 1
      for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
        gPackType vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FuncSumHalf, TF16>()(vals[u], vals2[u]);
      }

      if (COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 readVal;
          Fetch128(readVal, wPack+u*WARP_SIZE);
          Pack128 vval;
          Fetch128(vval, vPack+u*WARP_SIZE);
          Pack128 mval;
          Fetch128(mval, mPack+u*WARP_SIZE);

          Pack128 S5;
          uint2 __val = *(reinterpret_cast<const uint2*>(&vals[u]));
          Pack128 S4, S3, S2, S1;

          MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vval, __val, S2);
          MULTI128<binOp2<float>, half>().binOp2(beta2, S2, S4);

          MULTI128<mixedbinOp3<float>, half>().mixedbinOp1(beta2, mval, __val, S1);
          MULTI128<binOp2<float>, half>().binOp4(beta1, S1, S3);
          
          MULTI128<delta<float>, half>().delta(lr, S3, S4, S5);

          Store128(vPack+u*WARP_SIZE, S2);
          Store128(mPack+u*WARP_SIZE, S1);

          Pack128 finalVal;
          MULTI128<weightUpdate<float>, half>().weightUpdate(readVal, S5, finalVal);

  #if 0
          float2 h1 = __half22float2(*(reinterpret_cast<const half2*>(&__val.x)));
          if (h1.x != 4.0f) {
            gPackType v1 = *(srcs[0]+u*WARP_SIZE);
            gPackType v2 = *(srcs[1]+u*WARP_SIZE);

            half* hv1 = (half*)(&v1);
            half* hv2 = (half*)(&v2);

            printf("h1.x %f hv1 %f hv2 %f AUTOUNROLL %d elemOffset %d offset %d Npack %d Nelem %d MINSRCS %d nsrcs %d\n", h1.x, (float)hv1[0], (float)hv2[0], UNROLL, elemOffset, offset, Npack, Nelem, MINSRCS, nsrcs);

          }
  #endif

          // if (threadIdx.x == 0 && elemOffset == 0 && offset==0)   {
          //   float2 h1 = __half22float2(*(reinterpret_cast<const half2*>(&__val.x)));
          //   float2 h2 = __half22float2(*(reinterpret_cast<const half2*>(&__val.y)));
          //   float4 f4 = *(reinterpret_cast<float4*>(&finalVal));
          //   printf("f4.x %f f4.z %f f4.y %f f4.w %f h1.x %f h1.y %f h2.x %f h2.y %f MAXSRCS %d MINSRCS %d \n",f4.x, f4.z, f4.y, f4.w, h1.x, h1.y, h2.x, h2.y, MAXSRCS, MINSRCS);
          // }

          gPackType fp16FinalVal = float4ToHalf4(finalVal);
          Store128(wPack+u*WARP_SIZE, finalVal);

          *(dsts[0]+u*WARP_SIZE) = fp16FinalVal;

          for (int i = 1; i < MINDSTS; i++) {
            *(dsts[i]+u*WARP_SIZE) = fp16FinalVal;
          }

          #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            *(dsts[i]+u*WARP_SIZE) = fp16FinalVal;
          }
        }
      } else if (ALLGATHER_COMPUTE) {
        #if 0
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 readVal;
          Fetch128(readVal, dstW+u*WARP_SIZE);
          
          uint2 __val_delta = *(reinterpret_cast<const uint2*>(&vals[u]));
          // converterhalf1 ch;
          // ch.x0 = *(reinterpret_cast<const half2*>(&__val_delta.x));

          // converterhalf1 ch2;
          // ch2.x0 = *(reinterpret_cast<const half2*>(&__val_delta.y));

          // if (((float)ch.getx0()) != 1.0f) {
          //   printf("threadIdx.x %d v %f blockIdx.x %d\n", threadIdx.x, ((float)ch.getx0()), blockIdx.x);
          // }
          // assert(((float)ch.getx1()) == -1.0f);
          // assert(((float)ch2.getx0()) == -1.0f);
          // assert(((float)ch2.getx1()) == -1.0f);
          
          Pack128 finalVal;
          MULTI128<weightUpdate<float>, half>().weightUpdate(readVal, __val_delta, finalVal);

          Store128(dstW+u*WARP_SIZE, finalVal);

          for (int i = 1; i < MINDSTS; i++) {
            *(dsts[i]+u*WARP_SIZE) = vals[u];
          }

        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            *(dsts[i]+u*WARP_SIZE) = vals[u];
          }
        }

        #endif
      } 

  #if 0
      Pack128 vals[UNROLL];
      // Load and reduce
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);

      for (int i=1; i<MINSRCS; i++) {
        Pack128 vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) MULTI128<FuncSumHalf, TF16>()(vals[u], vals2[u]);
      }
      #pragma unroll 1
      for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
        Pack128 vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) MULTI128<FuncSumHalf, TF16>()(vals[u], vals2[u]);
      }
      // assert(COMPUTE == 1);
      if (COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          // uint2 __val1 = *(reinterpret_cast<const uint2*>(&vals[u].x));
          // uint2 __val2 = *(reinterpret_cast<const uint2*>(&vals[u].y));
          // converterhalf1 cS0;
          // converterhalf1 cS1;

          // cS0 = *(reinterpret_cast<const converterhalf1*>(&__val1.x));
          // cS1 = *(reinterpret_cast<const converterhalf1*>(&__val1.y));

          // if (__half2float(cS0.getx0()) != 2.0f) {
          //   printf("MINSRCS %d MAXSRCS %d nsrcs %d __half2float(cS0.getx0()) %f\n", MINSRCS, MAXSRCS, nsrcs, __half2float(cS0.getx0()));
          // }

          // assert(__half2float(cS0.getx0()) == 2.0f);
          // assert(__half2float(cS1.getx1()) == 2.0f);
          // assert(__half2float(cS0.getx0()) == 2.0f);
          // assert(__half2float(cS1.getx1()) == 2.0f);
  #if 0
          //Pack128 S2;
          Pack256 readVal256;
          Fetch256(readVal256, dsts[0]+u*WARP_SIZE);

          //First Pack128 of Pack256
          Pack128 S5x;
          Pack128 readValx = readVal256.x;
          uint2 __valx = *(reinterpret_cast<const uint2*>(&vals[u].x));

          MULTI128<mixedbinOp1<half>, half>().mixedbinOp1(beta2, readValx, __valx, S5x);

          //Second Part
          Pack128 S5y;
          Pack128 readValy = readVal256.y;
          uint2 __valy = *(reinterpret_cast<const uint2*>(&vals[u].y));

          MULTI128<mixedbinOp1<half>, half>().mixedbinOp1(beta2, readValy, __valy, S5y);

          Pack256 S5;
          S5.x = S5x;
          S5.y = S5y;
  #endif 

  #if 1
          //Pack128 S2;
          Pack256 readVal256;
          Fetch256(readVal256, dsts[0]+u*WARP_SIZE);
          Pack256 vval256;
          Fetch256(vval256, vPack+u*WARP_SIZE);
          Pack256 mval256;
          Fetch256(mval256, mPack+u*WARP_SIZE);

          //First Pack128 of Pack256
          Pack128 S5x;
          Pack128 readValx = readVal256.x;
          Pack128 vValx = vval256.x;
          Pack128 mValx = mval256.x;
          uint2 __valx = *(reinterpret_cast<const uint2*>(&vals[u].x));
          Pack128 S2x;
          Pack128 S1x;
          Pack128 S4, S3;

          MULTI128<mixedbinOp1<half>, half>().mixedbinOp1(beta2, vValx, __valx, S2x);
          MULTI128<binOp2<half>, half>().binOp2(beta2, S2x, S4);

          MULTI128<mixedbinOp3<half>, half>().mixedbinOp1(beta2, mValx, __valx, S1x);
          MULTI128<binOp2<half>, half>().binOp4(beta1, S1x, S3);
          
          MULTI128<delta<half>, half>().delta(lr, readValx, S3, S4, S5x);

          //Second Part
          Pack128 S5y;
          Pack128 readValy = readVal256.y;
          Pack128 vValy = vval256.y;
          Pack128 mValy = mval256.y;
          uint2 __valy = *(reinterpret_cast<const uint2*>(&vals[u].y));
          Pack128 S2y;
          Pack128 S1y;
          

          MULTI128<mixedbinOp1<half>, half>().mixedbinOp1(beta2, vValy, __valy, S2y);
          MULTI128<binOp2<half>, half>().binOp2(beta2, S2y, S4);

          MULTI128<mixedbinOp3<half>, half>().mixedbinOp1(beta2, mValy, __valy, S1y);
          MULTI128<binOp2<half>, half>().binOp4(beta1, S1y, S3);
          
          MULTI128<delta<half>, half>().delta(lr, readValy, S3, S4, S5y);

          Pack256 S5;
          S5.x = S5x;
          S5.y = S5y;
          
          Pack256 S2;
          S2.x = S2x;
          S2.y = S2y;

          Pack256 S1;
          S1.x = S1x;
          S1.y = S1y;

          Store256(vPack+u*WARP_SIZE, S2);
          Store256(mPack+u*WARP_SIZE, S1);
  #endif

          Store256(dsts[0]+u*WARP_SIZE, S5);

          for (int i = 1; i < MINDSTS; i++) {
            Store256(dsts[i]+u*WARP_SIZE, S5);
          }

          #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store256(dsts[i]+u*WARP_SIZE, S5);
          }

          // for (int i = 0; i < MINDSTS; i++) {
          //   Store128(dsts[i]+inc + u*WARP_SIZE, S5);
          // }

          // #pragma unroll 1
          // for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          //   Store128(dsts[i]+inc + u*WARP_SIZE, S5);
          // }
        }
      // } else if (ALLGATHER_COMPUTE) {
      //   for (int u = 0; u < UNROLL; ++u) {
      //     Pack128 readVal;
      //     Fetch128(readVal, dsts[0]+u*WARP_SIZE);
      //     Pack128 finalVal = vals[u];
          
      //     //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

      //     Store128(dsts[0]+u*WARP_SIZE, finalVal);

      //     for (int i = 0; i < MINDSTS; i++) {
      //       Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      //     }

      //   #pragma unroll 1
      //     for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
      //       Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      //     }
      //   }
      // } 
      } else {
        assert(false);
        // Store
        // for (int i = 0; i < MINDSTS; i++) {
        //   for (int u = 0; u < UNROLL; ++u) {
        //     uint2 __val = *(reinterpret_cast<const uint2*>(&vals[u]));
        //     converterhalf1 cS0;
        //     converterhalf1 cS1;

        //     cS0 = *(reinterpret_cast<const converterhalf1*>(&__val.x));
        //     cS1 = *(reinterpret_cast<const converterhalf1*>(&__val.y));

        //     converterfloat4toulong2 cVal;
        //     cVal.x0 = __half2float(cS0.getx0());
        //     cVal.x1 = __half2float(cS0.getx1());          
        //     cVal.x2 = __half2float(cS1.getx0());
        //     cVal.x3 = __half2float(cS1.getx1());

        //     Store128(dsts[i]+u*WARP_SIZE, cVal.storage);
        //   }
        // }
        // #pragma unroll 1
        // for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        //   for (int u = 0; u < UNROLL; ++u) {
        //     uint2 __val = *(reinterpret_cast<const uint2*>(&vals[u]));
        //     converterhalf1 cS0;
        //     converterhalf1 cS1;

        //     cS0 = *(reinterpret_cast<const converterhalf1*>(&__val.x));
        //     cS1 = *(reinterpret_cast<const converterhalf1*>(&__val.y));

        //     converterfloat4toulong2 cVal;
        //     cVal.x0 = __half2float(cS0.getx0());
        //     cVal.x1 = __half2float(cS0.getx1());          
        //     cVal.x2 = __half2float(cS1.getx0());
        //     cVal.x3 = __half2float(cS1.getx1());

        //     Store128(dsts[i]+u*WARP_SIZE, cVal.storage);
        //   }
        // }
      }
  #endif

      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      mPack += inc;
      vPack += inc;
      offset += inc;
      wPack += inc;
    }
  }

  template <typename T>
  __device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

  // Try to limit consecutive load/stores to 8.
  // Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
  #define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

  template<int UNROLL, class FUNC, typename TF16, typename TF32, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceOrCopyMultiComputation2(const int tid, const int nthreads,
      int nsrcs, const TF16* srcs[MAXSRCS], int ndsts, TF16* dsts[MAXDSTS],
      int N, TF32* w, TF32 lr, TF32 beta1, TF32 beta2, TF32* m, TF32* v) {
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
    
    int Npreamble = alignDiff ? Nrem :
      N < alignof(Pack128) ? N :
      (alignof(Pack128) - align) % alignof(Pack128);

    // stage 1: preamble: handle any elements up to the point of everything coming
    // into alignment
    if (Npreamble) {
      ReduceCopyMulti<FUNC, TF16, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
      Nrem -= Npreamble;
      if (Nrem == 0) return;
    }
    int offset = Npreamble;
    //assert(Npreamble == 0);
    // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit alignable.
    int wid = tid / WARP_SIZE;       // Warp number
    int nw = nthreads / WARP_SIZE; // Number of warps
    int t = tid % WARP_SIZE;       // Thread (inside the warp)

    const int packFactor = sizeof(Pack128) / sizeof(TF16);
    // stage 2a: main loop
    int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
        * (AUTOUNROLL * WARP_SIZE); // round down
    int Nelem2a = Npack2a * packFactor;

    ReduceCopy128bMultiComputation2<FUNC, TF16, TF32, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a, Nelem2a, w, lr, beta1, beta2, m, v);

    Nrem -= Nelem2a;
    if (Nrem == 0) return;
    offset += Nelem2a;
    // stage 2b: slightly less optimized for section when we don't have full
    // unrolling

    int Npack2b = Nrem / packFactor;
    int Nelem2b = Npack2b * packFactor;
  //assert(false);
    ReduceCopy128bMultiComputation2<FUNC, TF16, TF32, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b, Nelem2a, w, lr, beta1, beta2, m, v);

    Nrem -= Nelem2b;
    if (Nrem == 0) return;
    offset += Nelem2b;

    // stage 2c: tail
    //assert(false);
    ReduceCopyMulti<FUNC, TF16, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
  }

  template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceOrCopyMultiComputation(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      int N, T lr, T beta1, T beta2, T* m, T* v) {
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
    
    int Npreamble = alignDiff ? Nrem :
      N < alignof(Pack128) ? N :
      (alignof(Pack128) - align) % alignof(Pack128);

    // stage 1: preamble: handle any elements up to the point of everything coming
    // into alignment
    if (Npreamble) {
      ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
      Nrem -= Npreamble;
      if (Nrem == 0) return;
    }
    int offset = Npreamble;

    // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit alignable.
    int wid = tid / WARP_SIZE;       // Warp number
    int nw = nthreads / WARP_SIZE; // Number of warps
    int t = tid % WARP_SIZE;       // Thread (inside the warp)

    const int packFactor = sizeof(Pack128) / sizeof(T);

    // stage 2a: main loop
    int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
        * (AUTOUNROLL * WARP_SIZE); // round down
    int Nelem2a = Npack2a * packFactor;

    ReduceCopy128bMultiComputation<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a, lr, beta1, beta2, m, v);

    Nrem -= Nelem2a;
    if (Nrem == 0) return;
    offset += Nelem2a;

    // stage 2b: slightly less optimized for section when we don't have full
    // unrolling

    int Npack2b = Nrem / packFactor;
    int Nelem2b = Npack2b * packFactor;

    ReduceCopy128bMultiComputation<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b, lr, beta1, beta2, m, v);

    Nrem -= Nelem2b;
    if (Nrem == 0) return;
    offset += Nelem2b;

    // stage 2c: tail
    //assert(false);
    ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
  }

  template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
  __device__ __forceinline__ void ReduceCopy128bMulti( const int w, const int nw, const int t,
      int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
      const int elemOffset, const int Npack) {
    const int inc = nw * UNROLL * WARP_SIZE;
    int offset = w * UNROLL * WARP_SIZE + t;

    const Pack128* srcs[MAXSRCS];
    for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
    Pack128* dsts[MAXDSTS];
    for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

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


      if (false) {
        
      } else if (false) {
        
      } else {
        // Store
        for (int i = 0; i < MINDSTS; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
        #pragma unroll 1
        for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }

      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      offset += inc;
    }
  }

  // Try to limit consecutive load/stores to 8.
  // Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
  #define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

  template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
  __device__ __forceinline__ void ReduceOrCopyMulti(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      int N) {
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
    
    int Npreamble = alignDiff ? Nrem :
      N < alignof(Pack128) ? N :
      (alignof(Pack128) - align) % alignof(Pack128);

    // stage 1: preamble: handle any elements up to the point of everything coming
    // into alignment
    if (Npreamble) {
      ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
      Nrem -= Npreamble;
      if (Nrem == 0) return;
    }
    int offset = Npreamble;

    // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit alignable.
    int wid = tid / WARP_SIZE;       // Warp number
    int nw = nthreads / WARP_SIZE; // Number of warps
    int t = tid % WARP_SIZE;       // Thread (inside the warp)

    const int packFactor = sizeof(Pack128) / sizeof(T);

    // stage 2a: main loop
    int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
        * (AUTOUNROLL * WARP_SIZE); // round down
    int Nelem2a = Npack2a * packFactor;

    ReduceCopy128bMulti<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a);

    Nrem -= Nelem2a;
    if (Nrem == 0) return;
    offset += Nelem2a;

    // stage 2b: slightly less optimized for section when we don't have full
    // unrolling

    int Npack2b = Nrem / packFactor;
    int Nelem2b = Npack2b * packFactor;

    ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b);

    Nrem -= Nelem2b;
    if (Nrem == 0) return;
    offset += Nelem2b;

    // stage 2c: tail
    //assert(false);
    ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
  }

#elif TYPE_COMMON_KERNEL == 1

  // Define min for ssize_t
  static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

  typedef uint64_t PackType;

  // unpack x and y to elements of type T and apply FUNC to each element
  template<class FUNC, typename T>
  struct MULTI {
    __device__ PackType operator()(const PackType x, const PackType y) const;

    __device__ T atomicAdd(T* ptr, const PackType y) const;
      __device__ PackType binOp1(const  T beta2, const PackType v, const PackType S0) const;
      __device__ uint64_t mixedbinOp1(const  T beta2, const uint64_t v, const uint32_t S0) const;
      __device__ uint64_t mixedbinOp3(const  T beta1, const uint64_t m, const uint32_t S0) const;

    __device__ PackType binOp2(const  T beta2, const PackType S2) const;
    __device__ PackType binOp3(const  T beta1, const PackType m, const PackType S0) const;
    __device__ PackType binOp4(const  T beta1, const PackType S1) const;
    __device__ PackType binOp5(const  T lr, const PackType w, const PackType S3, const PackType S4) const;
    __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const;
      __device__ uint64_t delta(const  T ratio, const PackType rLambdaWeight) const;
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const;
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

    __device__ int8_t atomicAdd(int8_t* ptr, const PackType y) {}
    
    __device__ PackType binOp1(const int8_t beta2, const PackType v, const PackType S0) const {
  }
    __device__ PackType binOp2(const int8_t beta2, const PackType S2) const {
  }
    __device__ PackType binOp3(const int8_t beta1, const PackType m, const PackType S0) const {
  }
    __device__ PackType binOp4(const int8_t beta1, const PackType S1) const {
  }
    __device__ PackType binOp5(const int8_t lr, const PackType w, const PackType S3, const PackType S4) const {
  }
    __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
        __device__ uint64_t delta(const  int8_t lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
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

    __device__ uint8_t atomicAdd(uint8_t* ptr, const PackType y) {}

    
    __device__ PackType binOp1(const uint8_t beta2, const PackType v, const PackType S0) const {
  }
    __device__ PackType binOp2(const uint8_t beta2, const PackType S2) const {
  }
    __device__ PackType binOp3(const uint8_t beta1, const PackType m, const PackType S0) const {
  }
    __device__ PackType binOp4(const uint8_t beta1, const PackType S1) const {
  }
    __device__ PackType binOp5(const uint8_t lr, const PackType w, const PackType S3, const PackType S4) const {
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ uint64_t delta(const  uint8_t lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
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

    __device__ int32_t atomicAdd(int32_t* ptr, const PackType y) {
      converter cx;
      cx.storage = y;

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }
    
    
    __device__ PackType binOp1(const int32_t beta2, const PackType v, const PackType S0) const {
  converter cv;
  cv.storage = v;
  converter cS0;
  cS0.storage = S0;
  converter cS2;
  cS2.a = FUNC()(beta2, cv.a, cS0.a);
  cS2.b = FUNC()(beta2, cv.b, cS0.b);
  return cS2.storage;
  }
    __device__ PackType binOp2(const int32_t beta2, const PackType S2) const {
  converter cS2;
  cS2.storage = S2;
  converter cS4;
  cS4.a = FUNC()(beta2, cS2.a);
  cS4.b = FUNC()(beta2, cS2.b);
  return cS4.storage;
  }
    __device__ PackType binOp3(const int32_t beta1, const PackType m, const PackType S0) const {
  converter cm;
  cm.storage = m;
  converter cS0;
  cS0.storage = S0;
  converter cS1;
  cS1.a = FUNC()(beta1, cm.a, cS0.a);
  cS1.b = FUNC()(beta1, cm.b, cS0.b);
  return cS1.storage;
  }
    __device__ PackType binOp4(const int32_t beta1, const PackType S1) const {
  converter cS1;
  cS1.storage = S1;
  converter cS3;
  cS3.a = FUNC()(beta1, cS1.a);
  cS3.b = FUNC()(beta1, cS1.b);
  return cS3.storage;
  }
    __device__ PackType binOp5(const int32_t lr, const PackType w, const PackType S3, const PackType S4) const {
  converter cw;
  cw.storage = w;
  converter cS3;
  cS3.storage = S3;
  converter cS4;
  cS4.storage = S4;
  converter cS5;
  cS5.a = FUNC()(lr, cw.a, cS3.a, cS4.a);
  cS5.b = FUNC()(lr, cw.b, cS3.b, cS4.b);
  return cS5.storage;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ uint64_t delta(const  int32_t lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
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

    __device__ uint32_t atomicAdd(uint32_t* ptr, const PackType y) {
      converter cx;
      cx.storage = y;

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }

    
    __device__ PackType binOp1(const uint32_t beta2, const PackType v, const PackType S0) const {
  converter cv;
  cv.storage = v;
  converter cS0;
  cS0.storage = S0;
  converter cS2;
  cS2.a = FUNC()(beta2, cv.a, cS0.a);
  cS2.b = FUNC()(beta2, cv.b, cS0.b);
  return cS2.storage;
  }
    __device__ PackType binOp2(const uint32_t beta2, const PackType S2) const {
  converter cS2;
  cS2.storage = S2;
  converter cS4;
  cS4.a = FUNC()(beta2, cS2.a);
  cS4.b = FUNC()(beta2, cS2.b);
  return cS4.storage;
  }
    __device__ PackType binOp3(const uint32_t beta1, const PackType m, const PackType S0) const {
  converter cm;
  cm.storage = m;
  converter cS0;
  cS0.storage = S0;
  converter cS1;
  cS1.a = FUNC()(beta1, cm.a, cS0.a);
  cS1.b = FUNC()(beta1, cm.b, cS0.b);
  return cS1.storage;
  }
    __device__ PackType binOp4(const uint32_t beta1, const PackType S1) const {
  converter cS1;
  cS1.storage = S1;
  converter cS3;
  cS3.a = FUNC()(beta1, cS1.a);
  cS3.b = FUNC()(beta1, cS1.b);
  return cS3.storage;
  }
    __device__ PackType binOp5(const uint32_t lr, const PackType w, const PackType S3, const PackType S4) const {
  converter cw;
  cw.storage = w;
  converter cS3;
  cS3.storage = S3;
  converter cS4;
  cS4.storage = S4;
  converter cS5;
  cS5.a = FUNC()(lr, cw.a, cS3.a, cS4.a);
  cS5.b = FUNC()(lr, cw.b, cS3.b, cS4.b);
  return cS5.storage;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ uint64_t delta(const uint32_t lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
  };

  template<class FUNC>
  struct MULTI<FUNC, half> {
    static_assert(sizeof(PackType) == 4 * sizeof(half),
        "PackType must be four times the size of half.");

    struct PackHalf2 {
      half2 a, b;
    };

    __device__ PackType operator()(const PackType x, const PackType y) const {
      struct PackHalf2 cx, cy, cr;
      cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
      cy = *(reinterpret_cast<const struct PackHalf2*>(&y));

      cr.a = FUNC()(cx.a, cy.a);
      cr.b = FUNC()(cx.b, cy.b);

      return *(reinterpret_cast<PackType*>(&cr));
    }

      __device__ half atomicAdd(half* ptr, const PackType y) {
      // PackHalf2 cx;
      // cx = *(reinterpret_cast<const struct PackHalf2*>(&y));

      // ::atomicAdd(ptr, cx.a);
      // return ::atomicAdd(ptr, cx.b);
    }

    __device__ half2 atomicAdd(half2* ptr, const PackType y) {
      PackHalf2 cx;
      cx = *(reinterpret_cast<const struct PackHalf2*>(&y));

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }

      struct converterhalf{half2 x0;
    __device__ half getx0(){ return __low2half(x0);}
    __device__ half getx1(){ return __high2half(x0);}
  };
      union converterfloat{
        uint64_t storage;
        struct {float x0;
  __device__ float getx0(){ return x0;}
  float x1;
  __device__ float getx1(){ return x1;}
  ;}FOO;
  };
  __device__ uint64_t mixedbinOp1(const float beta2, const uint64_t v, const uint32_t S0) const {
  converterfloat cv;
  cv.storage = v;
  converterhalf cS0;
  cS0 = *(reinterpret_cast<const converterhalf*>(&S0));
  converterfloat cS2;
  cS2.FOO.x0 = FUNC()(beta2, cv.FOO.getx0(), cS0.getx0());
  cS2.FOO.x1 = FUNC()(beta2, cv.FOO.getx1(), cS0.getx1());
  //assert(cS2.FOO.x0 == 2.0f && cS2.FOO.x1 == 2.0f);
  return cS2.storage;
  }

  __device__ uint64_t mixedbinOp3(const float beta1, const uint64_t m, const uint32_t S0) const {
  converterfloat cm;
  cm.storage = m;
  converterhalf cS0;
  cS0 = *(reinterpret_cast<const converterhalf*>(&S0));
  converterfloat cS1;
  cS1.FOO.x0 = FUNC()(beta1, cm.FOO.getx0(), cS0.getx0());
  cS1.FOO.x1 = FUNC()(beta1, cm.FOO.getx1(), cS0.getx1());
  return cS1.storage;
  }

    __device__ PackType binOp1(const half beta2, const PackType v, const PackType S0) const {
  struct PackHalf2 cv;
  cv = *(reinterpret_cast<const struct PackHalf2*>(&v));
  struct PackHalf2 cS0;
  cS0 = *(reinterpret_cast<const struct PackHalf2*>(&S0));
  struct PackHalf2 cS2;
  cS2.a = FUNC()(__half2half2(beta2), cv.a, cS0.a);
  cS2.b = FUNC()(__half2half2(beta2), cv.b, cS0.b);
  return *(reinterpret_cast<PackType*>(&cS2));
  }
    __device__ PackType binOp2(const half beta2, const PackType S2) const {
  struct PackHalf2 cS2;
  cS2 = *(reinterpret_cast<const struct PackHalf2*>(&S2));
  struct PackHalf2 cS4;
  cS4.a = FUNC()(__half2half2(beta2), cS2.a);
  cS4.b = FUNC()(__half2half2(beta2), cS2.b);
  return *(reinterpret_cast<PackType*>(&cS4));
  }
    __device__ PackType binOp3(const half beta1, const PackType m, const PackType S0) const {
  struct PackHalf2 cm;
  cm = *(reinterpret_cast<const struct PackHalf2*>(&m));
  struct PackHalf2 cS0;
  cS0 = *(reinterpret_cast<const struct PackHalf2*>(&S0));
  struct PackHalf2 cS1;
  cS1.a = FUNC()(__half2half2(beta1), cm.a, cS0.a);
  cS1.b = FUNC()(__half2half2(beta1), cm.b, cS0.b);
  return *(reinterpret_cast<PackType*>(&cS1));
  }
    __device__ PackType binOp4(const half beta1, const PackType S1) const {
  struct PackHalf2 cS1;
  cS1 = *(reinterpret_cast<const struct PackHalf2*>(&S1));
  struct PackHalf2 cS3;
  cS3.a = FUNC()(__half2half2(beta1), cS1.a);
  cS3.b = FUNC()(__half2half2(beta1), cS1.b);
  return *(reinterpret_cast<PackType*>(&cS3));
  }
    __device__ PackType binOp5(const half lr, const PackType w, const PackType S3, const PackType S4) const {
  struct PackHalf2 cw;
  cw = *(reinterpret_cast<const struct PackHalf2*>(&w));
  struct PackHalf2 cS3;
  cS3 = *(reinterpret_cast<const struct PackHalf2*>(&S3));
  struct PackHalf2 cS4;
  cS4 = *(reinterpret_cast<const struct PackHalf2*>(&S4));
  struct PackHalf2 cS5;
  cS5.a = FUNC()(__half2half2(lr), cw.a, cS3.a, cS4.a);
  cS5.b = FUNC()(__half2half2(lr), cw.b, cS3.b, cS4.b);
  return *(reinterpret_cast<PackType*>(&cS5));
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ uint64_t delta(const  half lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
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

    __device__ float atomicAdd(float* ptr, const PackType y) {
      converter cx;
      cx.storage = y;

      ::atomicAdd(ptr, cx.a);
      return ::atomicAdd(ptr, cx.b);
    }

    
    __device__ PackType binOp1(const float beta2, const PackType v, const PackType S0) const {
  converter cv;
  cv.storage = v;
  converter cS0;
  cS0.storage = S0;
  converter cS2;
  cS2.a = FUNC()(beta2, cv.a, cS0.a);
  cS2.b = FUNC()(beta2, cv.b, cS0.b);
  return cS2.storage;
  }
    __device__ PackType binOp2(const float beta2, const PackType S2) const {
  converter cS2;
  cS2.storage = S2;
  converter cS4;
  cS4.a = FUNC()(beta2, cS2.a);
  cS4.b = FUNC()(beta2, cS2.b);
  return cS4.storage;
  }
    __device__ PackType binOp3(const float beta1, const PackType m, const PackType S0) const {
  converter cm;
  cm.storage = m;
  converter cS0;
  cS0.storage = S0;
  converter cS1;
  cS1.a = FUNC()(beta1, cm.a, cS0.a);
  cS1.b = FUNC()(beta1, cm.b, cS0.b);
  return cS1.storage;
  }
    __device__ PackType binOp4(const float beta1, const PackType S1) const {
  converter cS1;
  cS1.storage = S1;
  converter cS3;
  cS3.a = FUNC()(beta1, cS1.a);
  cS3.b = FUNC()(beta1, cS1.b);
  return cS3.storage;
  }
    __device__ PackType binOp5(const float lr, const PackType w, const PackType S3, const PackType S4) const {
  converter cw;
  cw.storage = w;
  converter cS3;
  cS3.storage = S3;
  converter cS4;
  cS4.storage = S4;
  converter cS5;
  cS5.a = FUNC()(lr, cw.a, cS3.a, cS4.a);
  cS5.b = FUNC()(lr, cw.b, cS3.b, cS4.b);
  return cS5.storage;
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
  __device__ uint64_t delta(const  float ratio, const PackType rLambdaWeight) const {
  converter crLambdaWeight;
  crLambdaWeight.storage = rLambdaWeight;
  converter cS5;
  cS5.a = FUNC()(ratio, crLambdaWeight.a);
  cS5.b = FUNC()(ratio, crLambdaWeight.b);
  return cS5.storage;
  }
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{
      converter cdelta;
  cdelta.storage = delta;
  converter cw;
  cw.storage = w;
  converter cS5;
  cS5.a = FUNC()(cw.a, cdelta.a);
  cS5.b = FUNC()(cw.b, cdelta.b);
  return cS5.storage;
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

    __device__ double atomicAdd(double* ptr, const PackType y) {
      return ::atomicAdd(ptr, __longlong_as_double(y));
    }

    
    __device__ PackType binOp1(const double beta2, const PackType v, const PackType S0) const {
      double rv = FUNC()(beta2, __longlong_as_double(v), __longlong_as_double(S0));
      return rv;
  }
    __device__ PackType binOp2(const double beta2, const PackType S2) const {
      double rv = FUNC()(beta2, __longlong_as_double(S2));
      return rv;
  }
    __device__ PackType binOp3(const double beta1, const PackType m, const PackType S0) const {
      double rv = FUNC()(beta1, __longlong_as_double(m), __longlong_as_double(S0));
      return rv;
  }
    __device__ PackType binOp4(const double beta1, const PackType S1) const {
      double rv = FUNC()(beta1, __longlong_as_double(S1));
      return rv;
  }
    __device__ PackType binOp5(const double lr, const PackType w, const PackType S3, const PackType S4) const {
      double rv = FUNC()(lr, __longlong_as_double(w), __longlong_as_double(S3), __longlong_as_double(S4));
      return rv;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ uint64_t delta(const  double lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
  };

  template<class FUNC>
  struct MULTI<FUNC, uint64_t> {
    static_assert(sizeof(PackType) == sizeof(uint64_t),
        "PackType must be the same size as uint64_t.");
    __device__ PackType operator()(const PackType x, const PackType y) const {
      uint64_t rv = FUNC()(x, y);
      return rv;
    }

    __device__ uint64_t atomicAdd(uint64_t* ptr, const PackType y) {
      return y;//::atomicAdd(ptr, y);
    }

    
    __device__ PackType binOp1(const uint64_t beta2, const PackType v, const PackType S0) const {
      uint64_t rv = FUNC()(beta2, v, S0);
      return rv;
  }
    __device__ PackType binOp2(const uint64_t beta2, const PackType S2) const {
      uint64_t rv = FUNC()(beta2, S2);
      return rv;
  }
    __device__ PackType binOp3(const uint64_t beta1, const PackType m, const PackType S0) const {
      uint64_t rv = FUNC()(beta1, m, S0);
      return rv;
  }
    __device__ PackType binOp4(const uint64_t beta1, const PackType S1) const {
      uint64_t rv = FUNC()(beta1, S1);
      return rv;
  }
    __device__ PackType binOp5(const uint64_t lr, const PackType w, const PackType S3, const PackType S4) const {
      uint64_t rv = FUNC()(lr, w, S3, S4);
      return rv;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ uint64_t delta(const  uint64_t lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
  };

  template<class FUNC>
  struct MULTI<FUNC, int64_t> {
    static_assert(sizeof(PackType) == sizeof(int64_t),
        "PackType must be the same size as int64_t.");
    __device__ PackType operator()(const PackType x, const PackType y) const {
      int64_t rv = FUNC()((int64_t)x, (int64_t)y);
      return rv;
    }

    __device__ int64_t atomicAdd(int64_t* ptr, const PackType y) {
      return y;//::atomicAdd(ptr, (int64_t)y);
    }

    
    __device__ PackType binOp1(const int64_t beta2, const PackType v, const PackType S0) const {
      int64_t rv = FUNC()(beta2, (int64_t)v, (int64_t)S0);
      return rv;
  }
    __device__ PackType binOp2(const int64_t beta2, const PackType S2) const {
      int64_t rv = FUNC()(beta2, (int64_t)S2);
      return rv;
  }
    __device__ PackType binOp3(const int64_t beta1, const PackType m, const PackType S0) const {
      int64_t rv = FUNC()(beta1, (int64_t)m, (int64_t)S0);
      return rv;
  }
    __device__ PackType binOp4(const int64_t beta1, const PackType S1) const {
      int64_t rv = FUNC()(beta1, (int64_t)S1);
      return rv;
  }
    __device__ PackType binOp5(const int64_t lr, const PackType w, const PackType S3, const PackType S4) const {
      int64_t rv = FUNC()(lr, (int64_t)w, (int64_t)S3, (int64_t)S4);
      return rv;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ uint64_t delta(const  int64_t lr, const uint64_t S3, const uint64_t S4) const{}
    __device__ uint64_t weightUpdate(const uint64_t w, uint64_t delta) const{}
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

    __device__ void mixedbinOp1(const float beta2, Pack128& v, uint2 S0, Pack128& S2) {
      S2.x = MULTI<FUNC, T>().mixedbinOp1(beta2, v.x, S0.x);
      S2.y = MULTI<FUNC, T>().mixedbinOp1(beta2, v.y, S0.y);
    }

    __device__ void mixedbinOp3(const float beta1, Pack128& m, uint2 S0, Pack128& S1) {
      S1.x = MULTI<FUNC, T>().mixedbinOp3(beta1, m.x, S0.x);
      S1.y = MULTI<FUNC, T>().mixedbinOp3(beta1, m.y, S0.y);
    }

    __device__ void binOp1(const float beta2, Pack128& v, Pack128& S0, Pack128& S2) {
      S2.x = MULTI<FUNC, T>().binOp1(beta2, v.x, S0.x);
      S2.y = MULTI<FUNC, T>().binOp1(beta2, v.y, S0.y);
    }
  __device__ void binOp2(const float beta2, Pack128& S2, Pack128& S4) {
      S4.x = MULTI<FUNC, T>().binOp2(beta2, S2.x);
      S4.y = MULTI<FUNC, T>().binOp2(beta2, S2.y);
    }
  __device__ void binOp3(const float beta1, Pack128& m, Pack128& S0, Pack128& S1) {
      S1.x = MULTI<FUNC, T>().binOp3(beta1, m.x, S0.x);
      S1.y = MULTI<FUNC, T>().binOp3(beta1, m.y, S0.y);
    }
  __device__ void binOp4(const float beta1, Pack128& S1, Pack128& S3) {
      S3.x = MULTI<FUNC, T>().binOp4(beta1, S1.x);
      S3.y = MULTI<FUNC, T>().binOp4(beta1, S1.y);
    }
  __device__ void binOp5(const float lr, Pack128& w, Pack128& S3, Pack128& S4, Pack128& S5) {
      S5.x = MULTI<FUNC, T>().binOp5(lr, w.x, S3.x, S4.x);
      S5.y = MULTI<FUNC, T>().binOp5(lr, w.y, S3.y, S4.y);
    }

    __device__ void r(Pack128& w, Pack128& S3, Pack128& S4, Pack128& S5) {
      S5.x = MULTI<FUNC, T>().r(w.x, S3.x, S4.x);
      S5.y = MULTI<FUNC, T>().r(w.y, S3.y, S4.y);
    }

    __device__ uint64_t delta(const T ratio, Pack128& rLambdaWeight, Pack128& out) const {
      out.x = MULTI<FUNC, T>().delta(ratio,rLambdaWeight.x);
      out.y = MULTI<FUNC, T>().delta(ratio,rLambdaWeight.y);
  }
    __device__ uint64_t weightUpdate(Pack128& w, Pack128& delta, Pack128& out) const{
      out.x = MULTI<FUNC, T>().weightUpdate(w.x,delta.x);
      out.y = MULTI<FUNC, T>().weightUpdate(w.y,delta.y);
    }
  };


  inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  }
  inline __device__ void Store128(Pack128* p, Pack128& v) {
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
  }

  template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
  __device__ __forceinline__ void ReduceCopyMulti(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      const int offset, const int N) {
    for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
      T val = vFetch(srcs[0]+idx);

      #pragma unroll
      for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
      #pragma unroll 1
      for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));

      #pragma unroll
      for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
      #pragma unroll 1
      for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
    }
  }

  template <typename T>
  __device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

  template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceCopy128bMultiComputation( const int w, const int nw, const int t,
      int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
      const int elemOffset, const int Npack, T lr, T beta1, T beta2, T* m, T* v, size_t startOffset, int partStartOffset, int partSize, T* rNorm, T* wNorm) {
    const int inc = nw * UNROLL * WARP_SIZE;
    int offset = w * UNROLL * WARP_SIZE + t;

    const Pack128* srcs[MAXSRCS];
    for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
    Pack128* dsts[MAXDSTS];
    for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;
      // Pack128* mPack = ((Pack128*)(m+elemOffset))+offset;
      // Pack128* vPack = ((Pack128*)(v+elemOffset))+offset;
    float reducedVal = 0.0;
    float wNormReduceVal = 0.0;
    assert (COMPUTE == 0);
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


      if (COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 readVal;
          Fetch128(readVal, dsts[0]+u*WARP_SIZE);
          float4 f4 = *(reinterpret_cast<float4*>(&readVal));
          wNormReduceVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;
          
          Pack128 finalVal = vals[u];
          // const size_t distributedSZ = (totalSize/nranks);
          // if (threadIdx.x == 0) {
          //   printf("nchunks %d chunk %d\n", nchunks, chunk);
          // }
          // const size_t perChannelDistributedSZ = distributedSZ/nchunks;
          // assert (perChannelDistributedSZ != 0);
          size_t mOffset = (startOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T)));//%(totalSize/nranks);
          //mOffset = partStartOffset + mOffset%partSize;
          // if (mOffset > 16384) {
          //   printf("mOffset %ld partSize %d partStartOffset %d Npack %d\n", mOffset, partSize, partStartOffset, Npack);
          // }
          if (ptrAlign128(m+mOffset) != 0) {
            //printf("mOffset %ld partSize %ld\n", mOffset, partSize);
          }
          Pack128* mPack = (Pack128*)(m+mOffset);
          Pack128* vPack = (Pack128*)(v+mOffset);
          Pack128 mPackVal;
          Fetch128(mPackVal, mPack);
          Pack128 vPackVal;
          Fetch128(vPackVal, vPack);
          Pack128 S2Pack;
          MULTI128<binOp1<T>, T>().binOp1(beta2, vPackVal, finalVal, S2Pack);
          Store128(vPack, S2Pack);
          Pack128 S4Pack;
          MULTI128<binOp2<T>, T>().binOp2(beta2, S2Pack, S4Pack);
          Pack128 S1Pack;
          MULTI128<binOp3<T>, T>().binOp3(beta1, mPackVal, finalVal, S1Pack);
          Store128(mPack, S1Pack);
          Pack128 S3Pack;
          MULTI128<binOp4<T>, T>().binOp4(beta1, S1Pack, S3Pack);
          // Pack128 S5Pack;
          // MULTI128<binOp5<T>, T>().binOp5(lr, readVal, S3Pack, S4Pack, finalVal);
          Pack128 S5Pack;
          MULTI128<rOp<T>, T>().r(readVal, S3Pack, S4Pack, finalVal);
          f4 = *(reinterpret_cast<float4*>(&finalVal));
          reducedVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;
          // ::atomicAdd((float*)(rNorm), f4.x*f4.x);
          // ::atomicAdd((float*)(rNorm), f4.y*f4.y);
          // ::atomicAdd((float*)(rNorm), f4.z*f4.z);
          // ::atomicAdd((float*)(rNorm), f4.w*f4.w);
          
          //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

          Store128(((Pack128*)srcs[0])+u*WARP_SIZE, finalVal);

        //   for (int i = 1; i < MINDSTS; i++) {
        //     Store128(dsts[i]+u*WARP_SIZE, finalVal);
        //   }

        // #pragma unroll 1
        //   for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        //     Store128(dsts[i]+u*WARP_SIZE, finalVal);
        //   }
        }
      } 
      // else if (ALLGATHER_COMPUTE) {
      //   for (int u = 0; u < UNROLL; ++u) {
      //     Pack128 readVal;
      //     Fetch128(readVal, dsts[0]+u*WARP_SIZE);
      //     Pack128 finalVal = vals[u];
          
      //     //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

      //     Store128(dsts[0]+u*WARP_SIZE, finalVal);

      //     for (int i = 0; i < MINDSTS; i++) {
      //       Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      //     }

      //   #pragma unroll 1
      //     for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
      //       Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      //     }
      //   }
      else {
        // Store
        for (int i = 0; i < MINDSTS; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
        #pragma unroll 1
        for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }

      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      offset += inc;
    }

    if (COMPUTE) {
      // for (int offset = warpSize/2; offset > 0; offset /= 2) 
      //   reducedVal += __shfl_down_sync(0xffffffff, reducedVal, offset);

      *((float*)rNorm) += reducedVal;
      // if (threadIdx.x % warpSize == 0)
      //   ::atomicAdd((float*)(rNorm), (float)(reducedVal)); 
      
      *((float*)wNorm) += wNormReduceVal;
    }
  }



  // Try to limit consecutive load/stores to 8.
  // Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
  #define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

  template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceOrCopyMultiComputation(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      int N, T lr, T beta1, T beta2, T* m, T* v, size_t startOffset, int partStartOffset, int partSize, T* rNorm, T* wNorm) {
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
    
    int Npreamble = alignDiff ? Nrem :
      N < alignof(Pack128) ? N :
      (alignof(Pack128) - align) % alignof(Pack128);

    // stage 1: preamble: handle any elements up to the point of everything coming
    // into alignment
    if (Npreamble) {
      ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
      Nrem -= Npreamble;
      if (Nrem == 0) return;
    }
    int offset = Npreamble;

    // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit alignable.
    int wid = tid / WARP_SIZE;       // Warp number
    int nw = nthreads / WARP_SIZE; // Number of warps
    int t = tid % WARP_SIZE;       // Thread (inside the warp)

    const int packFactor = sizeof(Pack128) / sizeof(T);

    // stage 2a: main loop
    int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
        * (AUTOUNROLL * WARP_SIZE); // round down
    int Nelem2a = Npack2a * packFactor;

    ReduceCopy128bMultiComputation<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a, lr, beta1, beta2, m, v, startOffset, partStartOffset, partSize, rNorm, wNorm);

    Nrem -= Nelem2a;
    if (Nrem == 0) return;
    offset += Nelem2a;

    // stage 2b: slightly less optimized for section when we don't have full
    // unrolling

    int Npack2b = Nrem / packFactor;
    int Nelem2b = Npack2b * packFactor;

    ReduceCopy128bMultiComputation<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b, lr, beta1, beta2, m, v, startOffset, partStartOffset, partSize, rNorm, wNorm);

    Nrem -= Nelem2b;
    if (Nrem == 0) return;
    offset += Nelem2b;

    // stage 2c: tail
    assert(false);
    //ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
  }

  struct FuncSumHalf {
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

  struct halfToUint64_t {
      half2 h1;
      half2 h2;
  };

  inline __device__ uint64_t float4ToHalf4(Pack128& v) {
    float2 h1 = *(reinterpret_cast<float2*>(&v.x));
    float2 h2 = *(reinterpret_cast<float2*>(&v.y));
    // assert (h1.x == -1.0f);
    // assert (h1.y == -1.0f);
    // assert (h1. == -1.0f);

    half2 r1 = __floats2half2_rn(h1.x, h1.y);
    half2 r2 = __floats2half2_rn(h2.x, h2.y);

    halfToUint64_t converter;
    converter.h1 = r1;
    converter.h2 = r2;

    return *(reinterpret_cast<uint64_t*>(&converter));
  }

  typedef uint64_t gPackType;

  template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceCopy128bMultiComputationMixedPrecision( const int w, const int nw, const int t,
      int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
      const int elemOffset, const int Npack, float lr, float beta1, float beta2, float* weight, float* m, float* v, float* r, size_t startOffset, int partStartOffset, int partSize, float* rNorm, float* wNorm) {
    const int inc = nw * UNROLL * WARP_SIZE;
    int offset = w * UNROLL * WARP_SIZE + t;

    const gPackType* srcs[MAXSRCS];
    for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const gPackType*)(s[i]+elemOffset))+offset;
    gPackType* dsts[MAXDSTS];
    for (int i=0; i<MAXDSTS; i++) dsts[i] = ((gPackType*)(d[i]+elemOffset))+offset;
      // Pack128* mPack = ((Pack128*)(m+elemOffset))+offset;
      // Pack128* vPack = ((Pack128*)(v+elemOffset))+offset;
    float reducedVal = 0.0;
    float wNormReduceVal = 0.0;
    assert(COMPUTE==1);

    while (offset < Npack*2) {
      gPackType vals[UNROLL];
      // Load and reduce
      for (int u = 0; u < UNROLL; ++u) vals[u] = *(srcs[0]+u*WARP_SIZE);

      for (int i=1; i<MINSRCS; i++) {
        gPackType vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FuncSumHalf, half>()(vals[u], vals2[u]);
      }
      #pragma unroll 1
      for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
        gPackType vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FuncSumHalf, half>()(vals[u], vals2[u]);
      }

      if (COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          size_t mOffset = (startOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(float)));//%(totalSize/nranks);
          Pack128 readVal;
          Pack128* weightPack = (Pack128*)(weight+mOffset);
          Pack128* rPack = (Pack128*)(r+mOffset);
          Pack128* mPack = (Pack128*)(m+mOffset);
          Pack128* vPack = (Pack128*)(v+mOffset);

          Fetch128(readVal, weightPack);
          float4 f4 = *(reinterpret_cast<float4*>(&readVal));
          // if ((f4.x != 1.0f)) {
          //   printf("f4.x %f mOffset %ld Npack %d startOffset %ld offset %d\n", f4.x, mOffset, Npack, startOffset, offset);
          // }
          //assert(f4.x == 1.0f);
          // assert(f4.y == 1.0f);
          // assert(f4.z == 1.0f);
          // assert(f4.w == 1.0f);

          // if (threadIdx.x == 0) {
            
          //   printf("f1 %f f2 %f f3 %f f4 %f\n", f4.x, f4.y, f4.z, f4.w);
          // }
          wNormReduceVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;
          
          uint2 __val = *(reinterpret_cast<const uint2*>(&vals[u]));

          Pack128 mPackVal;
          Fetch128(mPackVal, mPack);
          Pack128 vPackVal;
          Fetch128(vPackVal, vPack);
          Pack128 S2Pack;
          MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vPackVal, __val, S2Pack);
          Store128(vPack, S2Pack);
          Pack128 S4Pack;
          MULTI128<binOp2<float>, float>().binOp2(beta2, S2Pack, S4Pack);
          Pack128 S1Pack;
          MULTI128<mixedbinOp3<float>, half>().mixedbinOp3(beta1, mPackVal, __val, S1Pack);
          Store128(mPack, S1Pack);
          Pack128 S3Pack;
          MULTI128<binOp4<float>, float>().binOp4(beta1, S1Pack, S3Pack);
          // Pack128 S5Pack;
          // MULTI128<binOp5<T>, T>().binOp5(lr, readVal, S3Pack, S4Pack, finalVal);
          Pack128 finalVal;
          MULTI128<rOp<float>, float>().r(readVal, S3Pack, S4Pack, finalVal);
          f4 = *(reinterpret_cast<float4*>(&finalVal));
          reducedVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;
          Store128(rPack, finalVal);

          // ::atomicAdd((float*)(rNorm), f4.x*f4.x);
          // ::atomicAdd((float*)(rNorm), f4.y*f4.y);
          // ::atomicAdd((float*)(rNorm), f4.z*f4.z);
          // ::atomicAdd((float*)(rNorm), f4.w*f4.w);
          
          //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);
        //   for (int i = 1; i < MINDSTS; i++) {
        //     Store128(dsts[i]+u*WARP_SIZE, finalVal);
        //   }

        // #pragma unroll 1
        //   for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        //     Store128(dsts[i]+u*WARP_SIZE, finalVal);
        //   }
        }
      } 
      // else if (ALLGATHER_COMPUTE) {
      //   for (int u = 0; u < UNROLL; ++u) {
      //     Pack128 readVal;
      //     Fetch128(readVal, dsts[0]+u*WARP_SIZE);
      //     Pack128 finalVal = vals[u];
          
      //     //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

      //     Store128(dsts[0]+u*WARP_SIZE, finalVal);

      //     for (int i = 0; i < MINDSTS; i++) {
      //       Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      //     }

      //   #pragma unroll 1
      //     for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
      //       Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      //     }
      //   }
      else {
        // Store
        for (int i = 0; i < MINDSTS; i++) {
          for (int u = 0; u < UNROLL; ++u) 
          *(dsts[i]+u*WARP_SIZE) = vals[u];
        }
        #pragma unroll 1
        for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          for (int u = 0; u < UNROLL; ++u) 
          *(dsts[i]+u*WARP_SIZE) = vals[u];
        }
      }

      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      offset += inc;
    }

    if (COMPUTE) {
      // for (int offset = warpSize/2; offset > 0; offset /= 2) 
      //   reducedVal += __shfl_down_sync(0xffffffff, reducedVal, offset);

      *((float*)rNorm) += reducedVal;
      // if (threadIdx.x % warpSize == 0)
      //   ::atomicAdd((float*)(rNorm), (float)(reducedVal)); 
      
      *((float*)wNorm) += wNormReduceVal;
    }
  }



  // Try to limit consecutive load/stores to 8.
  // Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
  #define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

  template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceOrCopyMultiComputationMixedPrecision(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      int N, float lr, float beta1, float beta2, float* weight, float* m, float* v, float* r, size_t startOffset, int partStartOffset, int partSize, float* rNorm, float* wNorm) {
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
    
    int Npreamble = alignDiff ? Nrem :
      N < alignof(Pack128) ? N :
      (alignof(Pack128) - align) % alignof(Pack128);

    // stage 1: preamble: handle any elements up to the point of everything coming
    // into alignment
    if (Npreamble) {
      ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
      Nrem -= Npreamble;
      if (Nrem == 0) return;
    }
    int offset = Npreamble;

    // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit alignable.
    int wid = tid / WARP_SIZE;       // Warp number
    int nw = nthreads / WARP_SIZE; // Number of warps
    int t = tid % WARP_SIZE;       // Thread (inside the warp)

    const int packFactor = sizeof(Pack128) / sizeof(T);

    // stage 2a: main loop
    int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
        * (AUTOUNROLL * WARP_SIZE); // round down
    int Nelem2a = Npack2a * packFactor;

    ReduceCopy128bMultiComputationMixedPrecision<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a, lr, beta1, beta2, weight, m, v, r, startOffset, partStartOffset, partSize, rNorm, wNorm);

    Nrem -= Nelem2a;
    if (Nrem == 0) return;
    offset += Nelem2a;

    // stage 2b: slightly less optimized for section when we don't have full
    // unrolling

    int Npack2b = Nrem / packFactor;
    int Nelem2b = Npack2b * packFactor;

    ReduceCopy128bMultiComputationMixedPrecision<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b, lr, beta1, beta2, weight, m, v, r, startOffset, partStartOffset, partSize, rNorm, wNorm);

    Nrem -= Nelem2b;
    if (Nrem == 0) return;
    offset += Nelem2b;

    // stage 2c: tail
    assert(false);
    //ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
  }

  template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceCopy128bMultiComputationForComputeSend( const int w, const int nw, const int t,
      int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
      const int elemOffset, const int Npack, float lr, float beta1, float beta2, float* weight, float* m, float* v, float*r, size_t startOffset, int partStartOffset, int partSize, const  float rNorm, const float wNorm) {
    const int inc = nw * UNROLL * WARP_SIZE;
    int offset = w * UNROLL * WARP_SIZE + t;

    const gPackType* srcs[MAXSRCS];
    for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const gPackType*)(s[i]+elemOffset))+offset;
    gPackType* dsts[MAXDSTS];
    for (int i=0; i<MAXDSTS; i++) dsts[i] = ((gPackType*)(d[i]+elemOffset))+offset;

    while (offset < Npack*2) {
      gPackType vals[UNROLL];
      // Load and reduce
      // for (int u = 0; u < UNROLL; ++u) vals[u] = *(srcs[0]+u*WARP_SIZE);

      // for (int i=1; i<MINSRCS; i++) {
      //   gPackType vals2[UNROLL];
      //   for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
      //   for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FuncSumHalf, half>()(vals[u], vals2[u]);
      // }
      // #pragma unroll 1
      // for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
      //   gPackType vals2[UNROLL];
      //   for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
      //   for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FuncSumHalf, half>()(vals[u], vals2[u]);
      // }


      if (COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          size_t mOffset = (startOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(float)));//%(totalSize/nranks);
          Pack128* weightPack = (Pack128*)(weight+mOffset);
          Pack128* rPack = (Pack128*)(r+mOffset);
          
          Pack128 rLambdaW;
          Fetch128(rLambdaW, rPack);

          Pack128 weightVal;
          Fetch128(weightVal, weightPack);
          
          float scale = ((wNorm > 0) ? (rNorm > 0 ? wNorm/rNorm : 1.0f) : 1.0f)/rNorm;

          Pack128 finalVal;
          MULTI128<delta<float>, float>().delta(lr*scale, rLambdaW, finalVal);
          Pack128 vv;
          MULTI128<weightUpdate<float>, float>().weightUpdate(weightVal, finalVal, vv);
          // if (threadIdx.x == 0) {
          //   printf("scale %f wNorm '%f' rNorm '%f'\n", (float)scale, wNorm, rNorm);
          // }

          //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);
          // if (threadIdx.x == 0) {
          //   float* h4 = (float*)(&vv);
          //   printf("f1 %f f2 %f f3 %f f4 %f scale %f wNorm %f rNorm %f\n", h4[0], h4[1], h4[2], h4[3], scale, wNorm, rNorm);
          // }
          Store128(weightPack, vv);
          uint64_t halfWeightVal = float4ToHalf4(vv);
          *(((gPackType*)srcs[0]) + u*WARP_SIZE) = halfWeightVal;

          for (int i = 0; i < MINDSTS; i++) {
            *(dsts[i]+u*WARP_SIZE) = halfWeightVal;
          }

          #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            *(dsts[i]+u*WARP_SIZE) = halfWeightVal;
          }
        }
      } else if (false && ALLGATHER_COMPUTE) {
        // for (int u = 0; u < UNROLL; ++u) {
        //   Pack128 readVal;
        //   Fetch128(readVal, dsts[0]+u*WARP_SIZE);
        //   Pack128 finalVal = vals[u];
          
        //   //MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

        //   Store128(dsts[0]+u*WARP_SIZE, finalVal);

        //   for (int i = 0; i < MINDSTS; i++) {
        //     Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        //   }

        // #pragma unroll 1
        //   for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        //     Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        //   }
        // }
      } else {
        assert(false);
        // Store
        // for (int i = 0; i < MINDSTS; i++) {
        //   for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        // }
        // #pragma unroll 1
        // for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        //   for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        // }
      }

      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      offset += inc;
    }
  }


  template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ __forceinline__ void ReduceOrCopyMultiComputationForComputeSend(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      int N, float lr, float beta1, float beta2, float* weight, float* m, float* v, float* r, size_t startOffset, int partStartOffset, int partSize, float rNorm, float wNorm) {
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

    int Npreamble = alignDiff ? Nrem :
      N < alignof(Pack128) ? N :
      (alignof(Pack128) - align) % alignof(Pack128);

    // stage 1: preamble: handle any elements up to the point of everything coming
    // into alignment
    if (Npreamble) {
      ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
      Nrem -= Npreamble;
      if (Nrem == 0) return;
    }
    int offset = Npreamble;

    // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit alignable.
    int wid = tid / WARP_SIZE;       // Warp number
    int nw = nthreads / WARP_SIZE; // Number of warps
    int t = tid % WARP_SIZE;       // Thread (inside the warp)

    const int packFactor = sizeof(Pack128) / sizeof(T);

    // stage 2a: main loop
    int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
        * (AUTOUNROLL * WARP_SIZE); // round down
    int Nelem2a = Npack2a * packFactor;

    ReduceCopy128bMultiComputationForComputeSend<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a, lr, beta1, beta2, weight, m, v, r, startOffset, partStartOffset, partSize, rNorm, wNorm);

    Nrem -= Nelem2a;
    if (Nrem == 0) return;
    offset += Nelem2a;

    // stage 2b: slightly less optimized for section when we don't have full
    // unrolling

    int Npack2b = Nrem / packFactor;
    int Nelem2b = Npack2b * packFactor;

    ReduceCopy128bMultiComputationForComputeSend<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, COMPUTE, ALLGATHER_COMPUTE>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b, lr, beta1, beta2, weight, m, v, r, startOffset, partStartOffset, partSize, rNorm, wNorm);

    Nrem -= Nelem2b;
    if (Nrem == 0) return;
    offset += Nelem2b;

    // stage 2c: tail
    assert(false);
    //ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
  }

  template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
  __device__ __forceinline__ void ReduceCopy128bMulti( const int w, const int nw, const int t,
      int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
      const int elemOffset, const int Npack) {
    const int inc = nw * UNROLL * WARP_SIZE;
    int offset = w * UNROLL * WARP_SIZE + t;

    const Pack128* srcs[MAXSRCS];
    for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
    Pack128* dsts[MAXDSTS];
    for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

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


      if (false) {
        
      } else if (false) {
        
      } else {
        // Store
        for (int i = 0; i < MINDSTS; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
        #pragma unroll 1
        for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
          for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
        }
      }

      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      offset += inc;
    }
  }

  // Try to limit consecutive load/stores to 8.
  // Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
  #define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

  template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
  __device__ __forceinline__ void ReduceOrCopyMulti(const int tid, const int nthreads,
      int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
      int N) {
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
    
    int Npreamble = alignDiff ? Nrem :
      N < alignof(Pack128) ? N :
      (alignof(Pack128) - align) % alignof(Pack128);

    // stage 1: preamble: handle any elements up to the point of everything coming
    // into alignment
    if (Npreamble) {
      ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
      Nrem -= Npreamble;
      if (Nrem == 0) return;
    }
    int offset = Npreamble;

    // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit alignable.
    int wid = tid / WARP_SIZE;       // Warp number
    int nw = nthreads / WARP_SIZE; // Number of warps
    int t = tid % WARP_SIZE;       // Thread (inside the warp)

    const int packFactor = sizeof(Pack128) / sizeof(T);

    // stage 2a: main loop
    int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
        * (AUTOUNROLL * WARP_SIZE); // round down
    int Nelem2a = Npack2a * packFactor;

    ReduceCopy128bMulti<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a);

    Nrem -= Nelem2a;
    if (Nrem == 0) return;
    offset += Nelem2a;

    // stage 2b: slightly less optimized for section when we don't have full
    // unrolling

    int Npack2b = Nrem / packFactor;
    int Nelem2b = Npack2b * packFactor;

    ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(wid, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b);

    Nrem -= Nelem2b;
    if (Nrem == 0) return;
    offset += Nelem2b;

    // stage 2c: tail
    ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
  }

#endif 

#endif // COMMON_KERNEL_H_
