/**
    AD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUDATEST1_SRC_DATA_DATA_H_
#define CUDATEST1_SRC_DATA_DATA_H_

#include "../assert/Assert.h"
#include "../assert/GPUAssert.h"
#include "../misc/random.h"
#include "Array.h"
#include "CPUArray.h"
#include "GPUArray.h"
#include "Mode.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <new>
#include <ostream>
#include <random>

template<typename Type = float>
class SArray : public Array<Type> {

    using CPUArrayType = CPUArray<Type>;
    using GPUArrayType = GPUArray<Type>;

    using CPtr         = std::shared_ptr<CPUArrayType>;
    using GPtr         = std::shared_ptr<GPUArrayType>;

    public:
    // smart pointers to cpu and gpu values
    CPtr cpu_values = (CPtr) nullptr;
    GPtr gpu_values = (GPtr) nullptr;

    public:
    explicit SArray(ArraySizeType p_size) : Array<Type>(p_size) {}
    SArray(const SArray<Type>& other) : Array<Type>(other.m_size) {
        if (other.cpuIsAllocated()) {
            mallocCpu();
            this->template copyFrom<HOST>(other);
        }
        if (other.gpuIsAllocated()) {
            mallocGpu();
            this->template copyFrom<DEVICE>(other);
        }
    }
    SArray(SArray<Type>&& other) noexcept : Array<Type>(other.m_size) {
        this->cpu_values = other.cpu_values;
        this->gpu_values = other.gpu_values;
    }
    SArray<Type>& operator=(const SArray<Type>& other) {
        freeCpu();
        freeGpu();
        this->m_size = other.m_size;
        if (other.cpuIsAllocated()) {
            mallocCpu();
            this->template copyFrom<HOST>(other);
        }
        if (other.gpuIsAllocated()) {
            mallocGpu();
            this->template copyFrom<DEVICE>(other);
        }
        return (*this);
    }
    SArray<Type>& operator=(SArray<Type>&& other) noexcept {
        this->m_size     = std::move(other.m_size);
        this->cpu_values = std::move(other.cpu_values);
        this->gpu_values = std::move(other.gpu_values);
        return (*this);
    }
    virtual ~SArray() {
        freeCpu();
        freeGpu();
    }

    // allocate cpu and gpu memory functions
    template<Mode mode = HOST>
    void malloc() {
        if (isAllocated<mode>()) {
            free<mode>();
        }
        if constexpr (mode == HOST) {
            cpu_values = CPtr(new CPUArrayType(this->m_size));
        } else {
            gpu_values = GPtr(new GPUArrayType(this->m_size));
        }
    }
    void mallocCpu() { malloc<HOST>(); }
    void mallocGpu() { malloc<DEVICE>(); }

    // deallocate cpu and gpu memory
    template<Mode mode = HOST>
    void free() {
        if constexpr (mode == HOST) {
            cpu_values = nullptr;
        } else {
            gpu_values = nullptr;
        }
    }
    void freeCpu() { free<HOST>(); }
    void freeGpu() { free<DEVICE>(); }

    // checks if cpu and gpu memory is allocated
    template<Mode mode = HOST>
    bool isAllocated() const {
        if constexpr (mode == HOST) {
            return cpu_values != nullptr;
        } else {
            return gpu_values != nullptr;
        }
    }
    bool cpuIsAllocated() const { return isAllocated<HOST>(); }
    bool gpuIsAllocated() const { return isAllocated<DEVICE>(); }

    // returns the address of the cpu/gpu memory
    template<Mode mode = HOST>
    Type* address() const {
        if (isAllocated<mode>()) {
            if constexpr (mode == HOST) {
                return cpu_values->m_data;
            } else {
                return gpu_values->m_data;
            }
        }
        return nullptr;
    }
    Type* cpuAddress() const { return address<HOST>(); }
    Type* gpuAddress() const { return address<DEVICE>(); }

    // synchronise data between the cpu and the gpu memory only if both are allocated
    void gpuUpload() {
        if (!cpuIsAllocated() || !gpuIsAllocated())
            return;
        gpu_values->upload(*cpu_values.get());
    }
    void gpuDownload() {
        if (!cpuIsAllocated() || !gpuIsAllocated())
            return;
        gpu_values->download(*cpu_values.get());
    }

    // gets values from the cpu memory
    Type get(int height) const {
        ASSERT(cpuIsAllocated(), "cpu is not allocated for this array");
        ASSERT(height < this->size(), "invalid coordinate");
        ASSERT(height >= 0, "invalid coordinate");
        return cpu_values->m_data[height];
    }
    Type& get(int height) {
        ASSERT(cpuIsAllocated(), "cpu is not allocated for this array");
        ASSERT(height < this->size(), "invalid coordinate");
        ASSERT(height >= 0, "invalid coordinate");
        return cpu_values->m_data[height];
    }
    // operators to wrap the get functions
    Type  operator()(int height) const { return get(height); }
    Type& operator()(int height) { return get(height); }
    Type  operator[](int height) const { return get(height); }
    Type& operator[](int height) { return get(height); }

    // compute min max of cpu values
    [[nodiscard]] Type min() const {
        if (cpu_values == nullptr)
            return 0;
        Type m = get(0);
        for (int i = 0; i < this->size(); i++) {
            m = std::min(m, get(i));
        }
        return m;
    };
    [[nodiscard]] Type max() const {
        if (cpu_values == nullptr)
            return 0;
        Type m = get(0);
        for (int i = 0; i < this->size(); i++) {
            m = std::max(m, get(i));
        }
        return m;
    }
    [[nodiscard]] Type mean() const {
        Type res = 0;
        for (int i = 0; i < this->size(); i++) {
            res += this->get(i);
        }
        return res / this->size();
    }
    [[nodiscard]] Type std() const {
        Type res = 0;
        Type mea = mean();
        for (int i = 0; i < this->size(); i++) {
            res += std::pow((this->get(i) - mea), 2);
        }
        return std::sqrt(res / (this->size() - 1));
    }

    // sort values
    void sort() const {
        if (cpu_values == nullptr)
            return;
        std::sort(cpu_values->m_data, cpu_values->m_data + this->size(), std::greater<Type>());
    };

    // clear the given memory
    template<Mode mode = HOST>
    void clear() const {
        if (mode == HOST) {
            if (cpu_values == nullptr)
                return;
            cpu_values->clear();
        }
        if (mode == DEVICE) {
            if (gpu_values == nullptr)
                return;
            gpu_values->clear();
        }
    }

    // copy from another array
    template<Mode mode = HOST>
    void copyFrom(const SArray& other) {
        if constexpr (mode == HOST) {
            cpu_values->copyFrom(*other.cpu_values.get());
        } else if (mode == DEVICE) {
            gpu_values->copyFrom(*other.gpu_values.get());
        }
    }

    // create a new array based of the given slice.
    // start is inclusive, end is exclusive
    // only is performed on the cpu. GPU memory is not allocated
    SArray<Type> slice(int start, int end, int inc = 1) {

        if (start < 0)
            start = (start + this->size()) % this->size();
        if (end < start)
            end += this->size();

        if (inc > 0) {
            ArraySizeType size = std::ceil((end - start) / inc);
            SArray<Type>  res {size};
            res.mallocCpu();

            int idx = 0;
            for (int i = start; i < end; i += inc) {
                res[idx++] = this->get(i);
            }
            return std::move(res);
        }
        if (inc < 0) {
            ArraySizeType size = std::ceil((end - start) / -inc);
            SArray<Type>  res {size};
            res.mallocCpu();

            int idx = 0;
            for (int i = end - 1; i >= start; i += inc) {
                res[idx++] = this->get(i);
            }
            return std::move(res);
        }
        return SArray<Type> {0};
    }

    // randomisation
    void randomise(Type lower = 0, Type upper = 1) {
        if (cpu_values == nullptr)
            return;

        for (int i = 0; i < this->size(); i++) {
            this->get(i) = random::uniform(lower, upper);
        }
    }
    void randomiseGaussian(Type mean, Type deviation) {
        if (cpu_values == nullptr)
            return;
        for (int i = 0; i < this->size(); i++) {
            this->get(i) = random::normal(mean, deviation);
        }
    }

    // new instance
    [[nodiscard]] SArray<Type> newInstance() const { return SArray<Type> {this->size()}; }

    // output stream
    friend std::ostream& operator<<(std::ostream& os, const SArray& data) {
        os << "size:       " << data.size() << "\n"
           << "gpu_values: " << data.gpuAddress() << "\n"
           << "cpu_values: " << data.cpuAddress() << "\n";
        if (!data.cpuIsAllocated())
            return os;
        for (int n = 0; n < data.size(); n++) {
            os << std::fixed << std::setw(12) << std::right << std::setprecision(4)
               << (double) data.get(n);
        }
        os << "\n";
        return os;
    }
};

#endif    // CUDATEST1_SRC_DATA_DATA_H_
