/*

Copyright(c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#pragma once

#include <cmath>
#include <type_traits>
#include "aln.h"

namespace aln
{
    template<typename T>
    constexpr T MaxWeightValue() { return std::numeric_limits<T>::max(); }

    template<typename T>
    constexpr T MinWeightValue() { return -MaxWeightValue<T>(); }
    

    /// <summary>
    /// Defines constraints on a dimension of an ALN.
    /// </summary>
    /// <remarks>
    /// Instance members are not safe for use across multiple threads without 
    /// external synchronization.
    /// </remarks>
    template<typename T>
    struct DimensionalConstraint 
    {
    public:

        static_assert(std::is_floating_point_v<T>);

        static constexpr T MinWeightValue = MinWeightValue<T>();
        static constexpr T MaxWeightValue = MaxWeightValue<T>();

        DimensionalConstraint()
            : _minWeight(MinWeightValue)
            , _maxWeight(MaxWeightValue)
            , _epsilon(static_cast<T>(0.1))
        {
        }

        T MinWeight() const { return _minWeight; }
        void SetMinWeight(T value)
        {
            if (value != 0 && !std::isnormal(value))
                throw std::invalid_argument("invalid minimum weight");

            _minWeight = value;
        }

        T MaxWeight() const { return _maxWeight; }
        void SetMaxWeight(T value)
        {
            if (value != 0 && !std::isnormal(value))
                throw std::invalid_argument("invalid maximum weight");

            _maxWeight = value;
        }

        T Epsilon() const { return _epsilon; }
        void SetEpsilon(T value)
        {
            if (value <= 0 || !std::isnormal(value))
                throw std::invalid_argument("invalid epsilon");

            _epsilon = value;
        }

    private:
        T _minWeight;
        T _maxWeight;
        T _epsilon;
    };
}