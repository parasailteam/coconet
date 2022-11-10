/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SCATTEREDPRIMITIVES_H_
#define NCCL_SCATTEREDPRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
	if (SEND) { \
		/* Send to far first, then close */ \
		for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
		func(0, ##__VA_ARGS__); \
	} \
} while (0)

#define FOR_RECV(func, ...) do { \
	if (RECV) { \
		/* Recv from close first, then far */ \
		func(0, ##__VA_ARGS__); \
		for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
	} \
} while (0)

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
class ncclScatteredPrimitives {
	private:
		const int tid;
		const int nthreads;
		const int wid;
		const int wNumber;
		const int stepSize;
		const int maxPartSize;
  		const size_t totalSize;
  		const size_t nChannels;
  		const size_t loopSize;
  		const int nranks;
		const size_t nContiguousBuffs;
		
		// SCKL buffers
		// const ssize_t maxBuffSize;
		#define maxBuffSizeBits 10
		#define maxBuffSize (1<<maxBuffSizeBits)
		#define maxBitsForIndexWithinContiguousBuffer 25
		#define maxBitsForNumberOfBufferInContigeousBuffer 25
		const size_t* scatteredBuffSizes;
		const size_t nBuffs;
		const size_t* buffIdToParentBufferId;
		const size_t* parentBuffSizes;

		int nrecv = 0;
		int nsend = 0;
		struct ncclConnInfo* recvConn = NULL;
		volatile uint64_t* recvConnHeadPtr = NULL;
		uint64_t recvConnHead;
		volatile uint64_t* recvConnTailPtr = NULL;
		uint64_t recvConnTail;
		uint64_t recvConnTailCache; // Cache last seen value

		struct ncclConnInfo* sendConn = NULL;
		volatile int* sendConnFifoPtr = NULL;
		volatile uint64_t* sendConnTailPtr = NULL;
		uint64_t sendConnTail;
		volatile uint64_t* sendConnHeadPtr = NULL;
		uint64_t sendConnHead;
		uint64_t sendConnHeadCache; // Cache last seen value

		uint64_t recvStep[NRECV];
		uint64_t sendStep[NSEND];
		T** recvDirectBuff[NRECV];
		T** sendDirectBuff[NSEND];
		const T* recvBuff[NRECV];
		T* sendBuff[NSEND];
		struct ncclDevComm* comm;

		inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
		inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
		inline __device__ T* recvPtr(int i) { return ((T*)recvBuff[i])+recvOffset(i); }
		inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

		inline __device__ double* recvPtrDouble(int i) { return ((double*)recvBuff[i])+recvOffset(i)/(sizeof(double)/sizeof(T)); }
		inline __device__ double* sendPtrDouble(int i) { return ((double*)sendBuff[i])+sendOffset(i)/(sizeof(double)/sizeof(T)); }

		inline __device__ void barrier() {
			asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
		}
		inline __device__ void subBarrier() {
			asm volatile ("bar.sync 2, %0;" :: "r"(nthreads-WARP_SIZE));
		}

		uint32_t mismatch = 0;
		const uint64_t opCount;

		inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
			if (mismatch) {
				// In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
				*(comm->fatalDevError) = ncclDevAssertedMismatch;
			} else if (conn && *conn->opCountRem > opCount) {
				mismatch += 1;
			}
		}

		uint32_t spins = 0;
		uint32_t abort = 0;

		inline __device__ int checkAbort(int i, int send) {
			spins++;
			if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
				abort = *(comm->abortFlag);
				if (wid == i) checkMismatch(send ? sendConn : recvConn);
				spins = 0;
			}
			return abort;
		}

		inline __device__ void waitSend(int nbytes) {
			spins = 0;
			mismatch = 0;
			if (sendConnHeadPtr) {
				while (sendConnHeadCache + NCCL_STEPS < sendConnHead + SLICESTEPS) {
					sendConnHeadCache = *sendConnHeadPtr;
					if (checkAbort(wid, 1)) break;
				}
				if (sendConnFifoPtr) {
					sendConnFifoPtr[sendConnHead%NCCL_STEPS] = nbytes;
				}
				sendConnHead += SLICESTEPS;
			}
		}

		inline __device__ void waitRecv() {
			spins = 0;
			mismatch = 0;
			if (recvConnTailPtr) {
				while (recvConnTailCache < recvConnTail + SLICESTEPS) {
					recvConnTailCache = *recvConnTailPtr;
					if (checkAbort(wid, 0)) break;
				}
				recvConnTail += SLICESTEPS;
			}
		}

		inline __device__ ssize_t buffIndex(ssize_t offset){
			return offset / maxBuffSize;
		}

		inline __device__ int withinBuffOffset(ssize_t offset){
			return (int)(offset % maxBuffSize);
		}

		inline __device__ int BufferIndexWithinContiguousBuffer(ssize_t index){
			return ((scatteredBuffSizes[index] >> maxBuffSizeBits) & ((1<<maxBitsForIndexWithinContiguousBuffer) - 1));
		}

		inline __device__ int NumberOfBufferInContiguousBuffer(ssize_t index){
			return (scatteredBuffSizes[index] >> (maxBuffSizeBits+maxBitsForIndexWithinContiguousBuffer));
		}

		inline __device__ int BufferSize(ssize_t index){
			int size = scatteredBuffSizes[index] & (maxBuffSize-1);
			return (size == 0 ? maxBuffSize : size);
		}

		inline __device__ void incRecv(int i) {
			recvStep[i] += SLICESTEPS;
		}
		inline __device__ void postRecv() {
			if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += SLICESTEPS;
		}

		inline __device__ void incSend(int i) {
			sendStep[i] += SLICESTEPS;
		}
		inline __device__ void postSend() {
			if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += SLICESTEPS;
		}

		// template <int DIRECTRECV>
		// inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
		//   return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i][buffIndex(directOffset)]+withinBuffOffset(directOffset) : recvPtr(i);
		// }

		// template <int DIRECTSEND>
		// inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
		//   return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i][buffIndex(directOffset)]+withinBuffOffset(directOffset) : sendPtr(i);
		// }

		template <int DIRECTRECV>
		inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
			return DIRECTRECV && recvDirectBuff[i] ? ((const T*)recvDirectBuff[i])+directOffset : recvPtr(i);
		}

		template <int DIRECTSEND>
		inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
			return DIRECTSEND && sendDirectBuff[i] ? ((T*)sendDirectBuff[i])+directOffset : sendPtr(i);
		}

		template <int DIRECTRECV>
		inline __device__ const double* directRecvPtrDouble(int i, ssize_t directOffset) {
			return DIRECTRECV && recvDirectBuff[i] ? ((const double*)recvDirectBuff[i])+directOffset : recvPtrDouble(i);
		}

		template <int DIRECTSEND>
		inline __device__ double* directSendPtrDouble(int i, ssize_t directOffset) {
			return DIRECTSEND && sendDirectBuff[i] ? ((double*)sendDirectBuff[i])+directOffset : sendPtrDouble(i);
		}

		template <int DIRECTRECV>
			inline __device__ T* directRecvPtrIncScattered(int i, ssize_t thisBufferIndex, ssize_t thisWithinBufferOffset, T** baseRecvPtrs, ssize_t sliceAccumulation) {
				return DIRECTRECV && recvDirectBuff[i] ? (recvDirectBuff[i][thisBufferIndex]+thisWithinBufferOffset) : (baseRecvPtrs[i]+sliceAccumulation);
			}

		template <int DIRECTSEND>
			inline __device__ T* directSendPtrIncScattered(int i, ssize_t thisBufferIndex, ssize_t thisWithinBufferOffset, T** baseSendPtrs, ssize_t sliceAccumulation) {
				return DIRECTSEND && sendDirectBuff[i] ? (sendDirectBuff[i][thisBufferIndex]+thisWithinBufferOffset) : (baseSendPtrs[i]+sliceAccumulation);
			}

		template <int DIRECTRECV>
		inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
			return DIRECTRECV && recvDirectBuff[i] ? directInc : sliceInc;
		}

		template <int DIRECTSEND>
		inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
			return DIRECTSEND && sendDirectBuff[i] ? directInc : sliceInc;
		}

		template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
		inline __device__ void
		GenericOp(const double* srcPtr, double* dstPtr, int nelem, ssize_t directOffset) {
			int offset = 0;
			int sliceSize = stepSize/(sizeof(double)/sizeof(T))*SLICESTEPS;
			int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

			const double* srcs[RECV*NRECV+SRC];
			srcs[0] = SRC ? srcPtr : (const double*)directRecvPtrDouble<DIRECTRECV>(0, directOffset);
			if (RECV) {
			if (SRC) srcs[1] = (const double*)recvPtrDouble(0);
			for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = (const double*)recvPtrDouble(i);
			}

			double* dsts[SEND*NSEND+DST];
			dsts[0] = DST ? dstPtr : (double*)directSendPtrDouble<DIRECTSEND>(0, directOffset);
			if (SEND) {
			if (DST) dsts[1] = (double*)directSendPtrDouble<DIRECTSEND>(0, directOffset);
			for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = (double*)directSendPtrDouble<DIRECTSEND>(i, directOffset);
			}

			bool syncThread = tid >= nthreads-WARP_SIZE;

			#pragma unroll
			for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
			int realSize = max(0, min(dataSize, nelem-offset));
			if (!syncThread) {
				if (SEND) waitSend(realSize*sizeof(double));
				if (RECV) waitRecv();
				if (realSize > 0) {
				subBarrier();
				if (DIRECTRECV && recvDirectBuff[0]) {
					// We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
					if (SEND) {
					ReduceOrCopyMulti<UNROLL, FuncSum<double>, double, 1, 1, 1, NSEND,0,0,0,0>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, nullptr, nullptr, nullptr, realSize, 0,0,0,0,0,0,0,nullptr, nullptr, 0, nullptr);
					}
				} else {
					ReduceOrCopyMulti<UNROLL, FuncSum<double>, double, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST,0,0,0,0>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, nullptr, nullptr, nullptr,realSize, 0,0,0,0,0,0,0,nullptr,nullptr, 0, nullptr);
				}
				}
			}
			barrier();
			FOR_SEND(incSend);
			FOR_RECV(incRecv);
			if (syncThread) {
				if (SEND) {
				if (realSize > 0 && wid == 0) __threadfence_system();
				__syncwarp();
				postSend();
				}
				if (RECV) postRecv();
			}
			srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
			for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
			dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
			for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
			offset += realSize;
			}
		}
#define PRIMAUTOUNROLL (UNROLL*(4/(RECV+SRC+SEND+DST)))

		template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE>
			inline __device__ void
			GenericOpScattered(const T** srcPtr, T** dstPtr, T* firstMoment, T* secondMoment, T* rStorage, int nelem, ssize_t directOffset, const T alpha, 
					const T beta1, const T beta2, const int epoch, int partNum, double* weightNorm, double* rNorm) {
				int offset = 0;
				int sliceSize = stepSize*SLICESTEPS;
				int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

				ssize_t sliceAccumulation = 0;
				const T* srcs[RECV*NRECV+SRC];
				T* dsts[SEND*NSEND+DST];

				T* baseRecvPtrs[RECV*NRECV+SRC];
				T* baseSendPtrs[SEND*NSEND+DST];

				for (int i = 0; i < RECV*NRECV+SRC; i++){
					baseRecvPtrs[i] = recvPtr(i);
				}
				for (int i = 0; i < SEND*NSEND+DST; i++){
					baseSendPtrs[i] = sendPtr(i);
				}

				bool syncThread = tid >= nthreads-WARP_SIZE;
				// const int packFactor = sizeof(Pack128) / sizeof(T);
				size_t directOffsetBackup = (size_t)directOffset;
				int nActiveThreads = nthreads-WARP_SIZE;
				int nWarps = nActiveThreads / WARP_SIZE;
				const size_t slicedMomentSize = totalSize/nranks;
    			const int numParts = (maxPartSize > 0) ? max(1UL, slicedMomentSize/maxPartSize) : 1;
    			const int partsPerChannel = (totalSize < nranks * loopSize) ? 1 : DIVUP(totalSize, (nranks * loopSize));
    			const size_t partIdx = blockIdx.x*partsPerChannel + partNum;
				const size_t partStartOffset = partIdx*maxPartSize;

				#pragma unroll
				for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
					int realSize = max(0, min(dataSize, nelem-offset));
					if (!syncThread) {
						if (SEND) waitSend(realSize*sizeof(T));
						if (RECV) waitRecv();
						if (realSize > 0){
							subBarrier();
							ssize_t firstBufferIndex = buffIndex(directOffset);
							ssize_t lastBufferIndex = buffIndex(directOffset + realSize)-1;
							int firstWithinBufferOffset = withinBuffOffset(directOffset);
							int lastWithinBufferOffset = withinBuffOffset(directOffset+realSize);
							if (lastWithinBufferOffset)
								lastBufferIndex++;
							int lastBufferSize = BufferSize(lastBufferIndex);
							int lastBufferNumElems = min((lastWithinBufferOffset == 0 ?  maxBuffSize : lastWithinBufferOffset), lastBufferSize);
							
							ssize_t sizeProcessed = 0;
							ssize_t globalBuffIndex = firstBufferIndex + wNumber;
							while(realSize && globalBuffIndex <= lastBufferIndex) {
								int bufferIndexWithinContiguousBuffer = BufferIndexWithinContiguousBuffer(globalBuffIndex);
								int numberOfBufferInContiguousBuffer = NumberOfBufferInContiguousBuffer(globalBuffIndex);

								int minNumberOfBufferstoConsiderBefore = min(bufferIndexWithinContiguousBuffer, globalBuffIndex - firstBufferIndex);
								ssize_t firstContiguousBufferIndex = globalBuffIndex - minNumberOfBufferstoConsiderBefore;
								ssize_t lastContiguousBufferIndex = min(globalBuffIndex - bufferIndexWithinContiguousBuffer + numberOfBufferInContiguousBuffer-1, lastBufferIndex);
								int firstContiguousBufferOffset = ((firstContiguousBufferIndex == firstBufferIndex) ? firstWithinBufferOffset : 0);
								int lastContiguousBufferNumElems = ((lastContiguousBufferIndex == lastBufferIndex) ? lastBufferNumElems : BufferSize(lastContiguousBufferIndex));
								
								int relativeTid = minNumberOfBufferstoConsiderBefore * WARP_SIZE + wid;
								int sliceAccumulationOffset = (firstContiguousBufferIndex - firstBufferIndex) * maxBuffSize + firstContiguousBufferOffset - firstWithinBufferOffset;


								int nThreadsToConsider = WARP_SIZE * min((int)(lastContiguousBufferIndex - firstContiguousBufferIndex + 1), nWarps);
								int sizeToConsider = (lastContiguousBufferIndex - firstContiguousBufferIndex) * maxBuffSize - firstContiguousBufferOffset + lastContiguousBufferNumElems;

								// if (wid == 0 && blockIdx.x == 0 && WEIGHT_UPDATE && directOffset == 32171520){
								// 	printf("QQQ wNumber %d rId %d tId %d minNumberOfBufferstoConsiderBefore %d slice %d lastContiguousBufferIndex %d firstContiguousBufferIndex %d maxBuffSize %d firstContiguousBufferOffset %d BufferSize(lastContiguousBufferIndex) %d sizeToConsider %d, directOffset %d realSize %d SLICEPERCHUNK %d nelem %d, globalBuffIndex %d nWarps %d wNumber %d lastContiguousBufferNumElems = %d nThreadsToConsider = %d numberOfBufferInContiguousBuffer = %d nelem = %d\n", 
								// 		(int) wNumber, (int) relativeTid, (int) tid, (int) minNumberOfBufferstoConsiderBefore, (int) slice, (int) lastContiguousBufferIndex, (int) firstContiguousBufferIndex, (int) maxBuffSize, (int) firstContiguousBufferOffset, BufferSize(lastContiguousBufferIndex), sizeToConsider, (int) directOffset, (int) realSize, SLICESPERCHUNK, nelem, (int) globalBuffIndex, (int) nWarps, (int) wNumber, (int) lastContiguousBufferNumElems, (int) nThreadsToConsider, (int) numberOfBufferInContiguousBuffer, (int) nelem);
								// }
								if (sizeToConsider <= 0){
									break;
								}

								srcs[0] = SRC ? (srcPtr[firstContiguousBufferIndex]+firstContiguousBufferOffset) : directRecvPtrIncScattered<DIRECTRECV>(0, firstContiguousBufferIndex, firstContiguousBufferOffset, baseRecvPtrs, sliceAccumulation+sliceAccumulationOffset);
								if (RECV) {
									if (SRC) srcs[1] = baseRecvPtrs[0]+sliceAccumulation+sliceAccumulationOffset;
									for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = baseRecvPtrs[i]+sliceAccumulation+sliceAccumulationOffset;
								}

								dsts[0] = DST ? (dstPtr[firstContiguousBufferIndex]+firstContiguousBufferOffset) : directSendPtrIncScattered<DIRECTSEND>(0, firstContiguousBufferIndex, firstContiguousBufferOffset, baseSendPtrs, sliceAccumulation+sliceAccumulationOffset);
								if (SEND) {
									if (DST) dsts[1] = directSendPtrIncScattered<DIRECTSEND>(0, firstContiguousBufferIndex, firstContiguousBufferOffset, baseSendPtrs, sliceAccumulation+sliceAccumulationOffset);
									for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtrIncScattered<DIRECTSEND>(i, firstContiguousBufferIndex, firstContiguousBufferOffset, baseSendPtrs, sliceAccumulation+sliceAccumulationOffset);
								}

								//assert(buffIdToParentBufferId[firstContiguousBufferIndex] < nContiguousBuffs);

								T* firstMomentPtr = (WEIGHT_UPDATE || LAMB) ? (firstMoment) : NULL;
								T* secondMomentPtr = (WEIGHT_UPDATE || LAMB) ? (secondMoment) : NULL;
								if (sizeToConsider > 0) {
									int parentBuffId = 0;
									double* wNormPtr = nullptr;
									double* rNormPtr = nullptr;
									size_t parentBuffSize = 0;
									if (LAMB||LAMB_SEND_COMPUTE) {
										parentBuffId = buffIdToParentBufferId[firstContiguousBufferIndex];
										wNormPtr = &weightNorm[parentBuffId];
										rNormPtr = &rNorm[parentBuffId];
										parentBuffSize = parentBuffSizes[parentBuffId];
									}
									if (DIRECTRECV && recvDirectBuff[0]) {
										// We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
										if (SEND) {
											ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE, 0>(relativeTid, nThreadsToConsider, 1, srcs, nsend, dsts+1, firstMomentPtr, secondMomentPtr, rStorage, sizeToConsider, alpha, beta1, beta2, epoch, 
											// directOffsetBackup + offset+sizeProcessed, 
											firstContiguousBufferIndex*maxBuffSize+firstContiguousBufferOffset,
												partStartOffset, maxPartSize, wNormPtr, rNormPtr, parentBuffSize, nullptr);
										}
									} else {
										ssize_t oo = firstContiguousBufferIndex*maxBuffSize+firstContiguousBufferOffset;
										// if (parentBuffSizes[parentBuffId] ==  and oo < 31260672 + 2048) {
            							// 	printf("845: directOffset %ld firstContiguousBufferIndex %ld firstContiguousBufferOffset %ld oo %ld\n", directOffset + sizeProcessed, firstContiguousBufferIndex, firstContiguousBufferOffset, oo);
          								// }
										// if (LAMB_SEND_COMPUTE) {
										// 	if (oo >= 31260672) {
										// 		printf("wNorm %lf rNorm %lf\n", weightNorm[parentBuffId], rNorm[parentBuffId]);
										// 	}
										// }
										ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE, 0>(relativeTid, nThreadsToConsider, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, firstMomentPtr, secondMomentPtr, rStorage, sizeToConsider, alpha, beta1, beta2, epoch, 
										// directOffsetBackup + offset + sizeProcessed, 
										oo,
											partStartOffset, maxPartSize, wNormPtr, rNormPtr, parentBuffSize, nullptr);
									}
								}

								// subBarrier();
								// bool flag = true;
								// if (WEIGHT_UPDATE && wid == 0){
								// 	for (int k = 0; k < sizeToConsider; k++){
								// 		if (abs((float) dsts[0][k] - 1.0f) > 0.0001f || abs((float) dsts[1][k] - 1.0f) > 0.0001f){
								// 			printf("QQQ vals %f %f %d | firstBufferIndex %d bufferIndexWithinContiguousBuffer %d | wNumber %d rId %d tId %d minNumberOfBufferstoConsiderBefore %d slice %d lastContiguousBufferIndex %d firstContiguousBufferIndex %d maxBuffSize %d firstContiguousBufferOffset %d BufferSize(lastContiguousBufferIndex) %d sizeToConsider %d, directOffset %d realSize %d SLICEPERCHUNK %d nelem %d, globalBuffIndex %d nWarps %d wNumber %d lastContiguousBufferNumElems = %d nThreadsToConsider = %d numberOfBufferInContiguousBuffer = %d nelem = %d\n", 
								// 				(float) dsts[0][k], (float) dsts[1][k], k, (int) firstBufferIndex, (int) bufferIndexWithinContiguousBuffer, (int) wNumber, (int) relativeTid, (int) tid, (int) minNumberOfBufferstoConsiderBefore, (int) slice, (int) lastContiguousBufferIndex, (int) firstContiguousBufferIndex, (int) maxBuffSize, (int) firstContiguousBufferOffset, BufferSize(lastContiguousBufferIndex), sizeToConsider, (int) directOffset, (int) realSize, SLICESPERCHUNK, nelem, (int) globalBuffIndex, (int) nWarps, (int) wNumber, (int) lastContiguousBufferNumElems, (int) nThreadsToConsider, (int) numberOfBufferInContiguousBuffer, (int) nelem);
								// 			flag = false;
								// 		}
								// 	}
								// }
								// subBarrier();
								// if (flag == false)
								// 	assert(false);
								globalBuffIndex = firstBufferIndex + ((lastContiguousBufferIndex-firstBufferIndex - wNumber) / nWarps + 1)*nWarps +  wNumber;
								sizeProcessed += sizeToConsider;
							}
						}
					}

					barrier();
					FOR_SEND(incSend);
					FOR_RECV(incRecv);
					if (syncThread) {
						if (SEND) {
							if (realSize > 0 && wid == 0) __threadfence_system();
							__syncwarp();
							postSend();
						}
						if (RECV) postRecv();
					}
					sliceAccumulation += sliceSize;
					directOffset += realSize;
					// srcs[0] = SRC ? (srcPtr[buffIndex(directOffset)]+withinBuffOffset(directOffset)) : directRecvPtrInc<DIRECTRECV>(0, directOffset, sliceAccumulation);
					// if (RECV) {
					//  if (SRC) srcs[1] = recvPtr(0);
					//   for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i)+sliceAccumulation;
					// }
					// dsts[0] = DST ? (dstPtr[buffIndex(directOffset)]+withinBuffOffset(directOffset)) : directSendPtrInc<DIRECTSEND>(0, directOffset, sliceAccumulation);
					// if (SEND) {
					//   if (DST) directSendPtrInc<DIRECTSEND>(0, directOffset, sliceAccumulation);
					//   for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtrInc<DIRECTSEND>(i, directOffset, sliceAccumulation);
					// }
					// srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
					// for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
					// dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
					// for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
					offset += realSize;
					}
				}

				__device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T** directBuff) {
					recvBuff[i] = (const T*)conn->buff;
					recvStep[i] = conn->step;
					recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
					recvDirectBuff[i] = NULL;
					if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
						recvDirectBuff[i] = directBuff;
						if (tid == 0) *conn->ptrExchange = directBuff;
					}
					if (wid == i) recvConn = conn;
					if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
					nrecv++;
				}
				__device__ __forceinline__ void loadRecvSync() {
					if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
						recvConnTailPtr = recvConn->tail;
						recvConnTailCache = *recvConnTailPtr;
					}
					if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
						recvConnHeadPtr = recvConn->head;
						// Return credits in case we rounded up.
						*recvConnHeadPtr = recvConnHead;
						// Update opCount in case we skipped some operations
						*(recvConn->opCountLoc) = opCount;
					}
				}

				__device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i, T** directBuff) {
					sendBuff[i] = (T*)conn->buff;
					sendStep[i] = conn->step;
					sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
					sendDirectBuff[i] = NULL;
					if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
						void* volatile* ptr = conn->ptrExchange;
						while ((sendDirectBuff[i] = (T**)(*ptr)) == NULL);
						barrier();
						if (tid == 0) *ptr = NULL;
					}
					if (wid == i) sendConn = conn;
					if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
					nsend++;
				}
				__device__ __forceinline__ void loadSendSync() {
					if (tid < nsend) {
						sendConnHeadPtr = sendConn->head;
						sendConnHeadCache = *sendConnHeadPtr;
						sendConnFifoPtr = sendConn->fifo;
						*(sendConn->opCountLoc) = opCount;
					}
					if (tid >= nthreads-WARP_SIZE && wid<nsend) {
						sendConnTailPtr = sendConn->tail;
					}
				}

				__device__ __forceinline__ void saveRecvSync() {
					if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
						recvConn->step = recvConnHead;
						*(recvConn->opCountLoc) = opCount+1;
						__threadfence_system();
					}
				}

				__device__ __forceinline__ void saveSendSync() {
					if (tid < nsend) {
						sendConn->step = sendConnHead;
						*(sendConn->opCountLoc) = opCount+1;
						__threadfence_system();
					}
				}

				public:
				__device__ __forceinline__
					ncclScatteredPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T** directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, 
							const size_t* scatteredBuffSizes, const size_t nBuffs, const size_t totalSize, const int nranks, int nChannels, size_t loopSize, int maxPartSize, size_t nContiguousBuffs,
							const size_t* buffIdToParentBufferId, const size_t* parentBuffSizes)
					: comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), wNumber(tid/WARP_SIZE), stepSize(stepSize), opCount(opCount), scatteredBuffSizes(scatteredBuffSizes), nBuffs(nBuffs)
					, totalSize(totalSize), nranks(nranks), nChannels(nChannels), loopSize(loopSize), maxPartSize(maxPartSize), nContiguousBuffs(nContiguousBuffs),
					buffIdToParentBufferId(buffIdToParentBufferId), parentBuffSizes(parentBuffSizes) {
						// Make sure step is updated before we read it.
						barrier();
						for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
						for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, directBuff);
						loadRecvSync();
						loadSendSync();
					}

				__device__ __forceinline__ void
					sendScattered(const T** src, ssize_t directOffset, int nelem) {
						GenericOpScattered<0, 0, 0, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nullptr,nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, 0, 0, nullptr, nullptr);
					}
				__device__ __forceinline__ void
					directSendScattered(const T** src, ssize_t directOffset, int nelem) {
						GenericOpScattered<0, 1, 0, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nullptr,nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, 0, 0, nullptr, nullptr);
					}
					
					__device__ __forceinline__ void
					directSendScatteredLAMB(const T** src, T* rStorage, ssize_t directOffset, int nelem, int partNum, T lr, double* wNorm, double* rNorm) {
						GenericOpScattered<0, 1, 0, 1, 1, 0, 0, 0, 1>(src, nullptr, nullptr, nullptr, rStorage,nelem, directOffset, lr, (T)0.0, (T)0.0, 0, partNum, wNorm, rNorm);
					}

				__device__ __forceinline__ void
					recvScattered(T** dst, int nelem) {
						GenericOpScattered<0, 0, 1, 0, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nullptr,nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0,  0, nullptr, nullptr);
					}
				__device__ __forceinline__ void
					directRecvScattered(T** dst, ssize_t directOffset, int nelem) {
						GenericOpScattered<1, 0, 1, 0, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nullptr,nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}

				__device__ __forceinline__ void
					copySendScattered(const T** src, T* dst, int nelem) {
						GenericOpScattered<0, 0, 0, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nullptr,nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}
				__device__ __forceinline__ void
					directCopySendScattered(const T** src, T* dst, ssize_t directOffset, int nelem) {
						GenericOpScattered<0, 1, 0, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nullptr,nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}

				__device__ __forceinline__ void
					recvCopySendScattered(T** dst, int nelem) {
						GenericOpScattered<0, 0, 1, 1, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nullptr,nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}
				__device__ __forceinline__ void
					directRecvCopySendScattered(T** dst, ssize_t directOffset, int nelem) {
						GenericOpScattered<1, 1, 1, 1, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nullptr,nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}

				__device__ __forceinline__ void
					recvReduceCopyScattered(const T** src, T** dst, int nelem) {
						GenericOpScattered<0, 0, 1, 0, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nullptr,nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}

				__device__ __forceinline__ void
					recvReduceSendScattered(const T** src, ssize_t directOffset, int nelem) {
						GenericOpScattered<0, 0, 1, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nullptr,nelem,directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}

				__device__ __forceinline__ void
					recvReduceCopySendScattered(const T** src, T** dst, int nelem) {
						GenericOpScattered<0, 0, 1, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nullptr,nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}
				__device__ __forceinline__ void
					directRecvReduceCopySend(const T** src, T** dst, ssize_t directOffset, int nelem) {
						// Direct is only for the send part
						GenericOpScattered<0, 1, 1, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL,nullptr, nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}

				__device__ __forceinline__ void 
					directRecvReduceCopySendAdam(const T** src, T** weight, T* firstMoment, T* secondMoment, 
							ssize_t directOffset, int nelem, const T alpha, const T beta1, const T beta2, const int epoch, int partNum) {
						GenericOpScattered<0, 1, 1, 1, 1, 1, 1, 0, 0>(src, weight, firstMoment, secondMoment, nullptr, nelem, directOffset, alpha, beta1, beta2, epoch, partNum, nullptr, nullptr);
					}

					__device__ __forceinline__ void 
					directRecvReduceCopyLAMB(const T** src, T** weight, T* firstMoment, T* secondMoment, T* rStorage,
							ssize_t directOffset, int nelem, const T alpha, const T beta1, const T beta2, const int epoch, int partNum, double* weightNorm, double* rNorm) {
						GenericOpScattered<0, 0, 1, 0, 1, 1, 1, 1, 0>(src, weight, firstMoment, secondMoment, rStorage, nelem, directOffset, alpha, beta1, beta2, epoch, partNum, weightNorm, rNorm);
					}

				__device__ __forceinline__ void 
					directRecvReduceCopySendWeight(const T** src, T** weight, ssize_t directOffset, int nelem, const T alpha) {
						GenericOpScattered<0, 1, 1, 1, 1, 1, 1, 0, 0>(src, weight, NULL, NULL, nullptr, nelem, directOffset, alpha, (T)0.0, (T)0.0, (T)0.0, 0, nullptr, nullptr);
					}

				  __device__ __forceinline__ void
					send(const double* src, int nelem) {
						GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
					}
					__device__ __forceinline__ void
					directSend(const double* src, ssize_t directOffset, int nelem) {
						GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
					}

					__device__ __forceinline__ void
					recv(double* dst, int nelem) {
						GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
					}
					__device__ __forceinline__ void
					directRecv(double* dst, ssize_t directOffset, int nelem) {
						GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
					}

					__device__ __forceinline__ void
					copySend(const double* src, double* dst, int nelem) {
						GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
					}
					__device__ __forceinline__ void
					directCopySend(const double* src, double* dst, ssize_t directOffset, int nelem) {
						GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
					}

					__device__ __forceinline__ void
					recvCopySend(double* dst, int nelem) {
						GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
					}
					__device__ __forceinline__ void
					directRecvCopySend(double* dst, ssize_t directOffset, int nelem) {
						GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
					}

					__device__ __forceinline__ void
					recvReduceCopy(const double* src, double* dst, int nelem) {
						GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
					}

					__device__ __forceinline__ void
					recvReduceSend(const double* src, int nelem) {
						GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
					}

					__device__ __forceinline__ void
					recvReduceCopySend(const double* src, double* dst, int nelem) {
						GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
					}
					__device__ __forceinline__ void
					directRecvReduceCopySend(const double* src, double* dst, ssize_t directOffset, int nelem) {
						// Direct is only for the send part
						GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
					}

				__device__ __forceinline__ ~ncclScatteredPrimitives() {
					// Save steps for the next operation
					saveRecvSync();
					saveSendSync();
				}
			};

		// #include "prims_ll.h"
		// #include "scattered_prims_ll.h"
		//#include "prims_ll128.h"

#endif
