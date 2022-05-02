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

#include <limits>

#include "aln.h"

namespace aln
{
    template<typename T>
    class Network;

    /// <summary>
    /// Represents a linear function.
    /// </summary>
    /// <remarks>
    /// Instance members are not safe for use across multiple threads without 
    /// external synchronization.
    /// </remarks>
    template<typename T>
    class LinearUnit final
    {
    public:

        /// <summary>
        /// Constructs a new instance of <see cref="LinearUnit"/> suitable for deserialization via move assignment.
        /// </summary>
        LinearUnit()
            : _isConstant{ false }
            , _isSplitAllowed{ false }
            , _parentIndex{ }
            , _bias{ }
        {
        }

        /// <summary>
        /// Constructs an instance of a <see cref="MinMaxNode"/>.
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="parentIndex"></param>
        LinearUnit(size_t dimensions,  NodeIndex parentIndex)
            : _isConstant{ false }
            , _isSplitAllowed{ false }
            , _parentIndex{ parentIndex }
            , _bias{ }
            , _w(dimensions)
            , _c(dimensions + 1)
            , _d(dimensions)
        {
        }

        /// <summary>
        /// Get index of parent in the network.
        /// </summary>
        /// <remarks>
        /// If ParentIndex().IsRoot() == true, then this node is the root and refers to itself.
        /// </remarks>
        NodeIndex ParentIndex() const
        {
            return _parentIndex;
        }

        bool IsConstant() const { return _isConstant; }
        void SetIsConstant(bool value)
        {
            if (_w.size() == 0)
                throw std::logic_error("zero dimensional linear unit");

            _isConstant = value;
            if (_isConstant)
                _isSplitAllowed = false;
        }

        void SetConstantValue(
            T value, 
            T d = static_cast<T>(0.0000001))
        {
            if (_w.size() == 0)
                throw std::logic_error("zero dimensional linear unit");

            if (d <= 0)
                throw std::invalid_argument("d must be positive");

            _isConstant = true;
            _isSplitAllowed = false;

            auto const dimensions = _w.size();

            std::fill_n(begin(_w), dimensions, static_cast<T>(0));
            std::fill_n(begin(_c), dimensions, static_cast<T>(0));
            std::fill_n(begin(_d), dimensions, d);

            // set bias and centroid at output to the constant
            _bias = value;
            _c[dimensions] = value;
        }

        bool IsSplitAllowed() const { return _isSplitAllowed; }
        void SetIsSplitAllowed(bool value)
        {
            if (_w.size() == 0)
                throw std::logic_error("zero dimensional linear unit");

            _isSplitAllowed = value;
            if (_isSplitAllowed)
                _isConstant = false;
        }

        // bias
        T Bias() const { return _bias; }
        void SetBias(T const value) { _bias = value; }

        // weight vector - Dimensions() elements
        FixedVector<T>& W() { return _w; }
        FixedVector<T> const& W() const { return _w; }

        // centroid - Dimensions() + 1 elements
        FixedVector<T>& C() { return _c; }
        FixedVector<T> const& C() const { return _c; }

        // average distance from centroid - Dimensions() elements
        FixedVector<T>& D() { return _d; }
        FixedVector<T> const& D() const { return _d; }

    private:

        friend class Network<T>;

        void ReplaceParent(NodeIndex parentIndex)
        {
            assert(!parentIndex.IsLinear() && !parentIndex.IsInvalid());
            _parentIndex = parentIndex;
        }

        bool _isConstant;
        bool _isSplitAllowed;
        
        NodeIndex _parentIndex;
        
        T _bias;
        FixedVector<T> _w;

        FixedVector<T> _c;
        FixedVector<T> _d;
    };
}

