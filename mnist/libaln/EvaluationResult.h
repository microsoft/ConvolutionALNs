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

#include <stack>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <type_traits>

#include "aln.h"

namespace aln
{
    /// <summary>
    /// Holds the result of an evaluation.
    /// </summary>
    template <typename T>
    class EvaluationResult final
    {
    public:

        static_assert(std::is_floating_point_v<T>);
        constexpr static T NaN = NaN<T>();

        EvaluationResult()
            : _value { NaN }
            , _surfaceResponsibility{}
        {
        }

        /// <summary>
        /// Resets all values to an unevaluated state and reserves enough memory to evaluate the specified network.
        /// </summary>
        /// <param name="aln"></param>
        void Reset(Network<T> const& aln)
        {
            auto const minMaxCount = aln.MinMaxNodes().size();
            auto const linearCount = aln.LinearUnits().size();

            _responsibleChildren.resize(minMaxCount);
            _minMaxValues.resize(minMaxCount);
            _linearValues.resize(linearCount);

            std::fill_n(begin(_responsibleChildren), minMaxCount, NodeIndex::Invalid());
            std::fill_n(begin(_minMaxValues), minMaxCount, NaN);
            std::fill_n(begin(_linearValues), linearCount, NaN);

            _value = NaN;
            _surfaceResponsibility = NodeIndex::Invalid();
        }

        /// <summary>
        /// Clears all memory held by the result and sets all values to an unevaluated state.
        /// </summary>
        void Clear()
        {
            _responsibleChildren.resize(0);
            _minMaxValues.resize(0);
            _linearValues.resize(0);

            _value = NaN;
            _surfaceResponsibility = NodeIndex::Invalid();
        }

        T Value() const { return _value; }
        void SetValue(T value) { _value = value; }

        NodeIndex SurfaceResponsibility() const { return _surfaceResponsibility; }
        void SetSurfaceResponsibility(NodeIndex nodeIndex) 
        { 
            if (nodeIndex.IsInvalid() || nodeIndex.IsMinMax())
                throw std::invalid_argument("invalid node index");

            _surfaceResponsibility = nodeIndex; 
        }

        NodeIndex ResponsibleChildOf(NodeIndex nodeIndex) const
        {
            if (nodeIndex.IsInvalid() || nodeIndex.IsLinear())
                throw std::invalid_argument("invalid node index");

            return _responsibleChildren[nodeIndex.Ordinal()];
        }

        T ValueOf(NodeIndex nodeIndex) const
        {
            if (nodeIndex.IsInvalid())
                throw std::invalid_argument("node index is not valid");

            return nodeIndex.IsLinear()
                ? _linearValues[nodeIndex.Ordinal()]
                : _minMaxValues[nodeIndex.Ordinal()];
        }

        void SetLinearResult(NodeIndex nodeIndex, T const value)
        {
            if (nodeIndex.IsInvalid() || nodeIndex.IsMinMax())
                throw std::invalid_argument("node index is not valid");

            _linearValues[nodeIndex.Ordinal()] = value;
        }

        void SetMinMaxResult(NodeIndex nodeIndex, std::tuple<NodeIndex, T> const& result)
        {
            if (nodeIndex.IsInvalid() || nodeIndex.IsLinear())
                throw std::invalid_argument("node index is not valid");

            std::tie(_responsibleChildren[nodeIndex.Ordinal()], _minMaxValues[nodeIndex.Ordinal()]) = result;
        }

        size_t TotalEvaluations() const 
        { 
            return TotalLinearEvaluations() + TotalMinMaxEvaluations();
        }

        size_t TotalLinearEvaluations() const
        {
            return std::count_if(
                Sequential,
                begin(_linearValues),
                end(_linearValues),
                IsComputed<T>);
        }

        size_t TotalMinMaxEvaluations() const
        {
            return std::count_if(
                Sequential,
                begin(_minMaxValues),
                end(_minMaxValues),
                IsComputed<T>);
        }

        std::vector<NodeIndex> const& ResponsibleChildren() const { return _responsibleChildren; }
        std::vector<T> const& MinMaxValues() const { return _minMaxValues; }
        std::vector<T> const& LinearValues() const { return _linearValues; }

    private:

        T _value;
        NodeIndex _surfaceResponsibility;

        std::vector<NodeIndex> _responsibleChildren;
        std::vector<T> _minMaxValues;
        std::vector<T> _linearValues;
    };

    /// <summary>
    /// A thread-safe pool of evaluation results that can be used during evaluation of many data points at once.
    /// Requests that would exceed the capacity of the pool will block until a previous result is returned.
    /// </summary>
    template <typename T>
    class EvaluationResultPool final
    {
    public:

        inline static const size_t DefaultCapacity = std::thread::hardware_concurrency() * 2;

        EvaluationResultPool(size_t capacity = DefaultCapacity)
            : _capacity{ static_cast<size_t>(std::max<size_t>(capacity, 1)) }
        {
            for (size_t i = 0U; i < _capacity; i++)
            {
                _pool.emplace();
            }
        }

        size_t Capacity() const
        {
            return _capacity;
        }

        EvaluationResult<T> Acquire()
        {
            std::unique_lock<std::mutex> lock(_mutex);

            _condition.wait(lock, [&]()
                {
                    return !_pool.empty();
                });
            
            auto result = std::move(_pool.top());
            _pool.pop();

            return result;
        }

        void Release(EvaluationResult<T> result)
        {
            std::unique_lock<std::mutex> lock(_mutex);
            
            if (_pool.size() == _capacity)
                throw std::logic_error("cannot return result to a full pool.");

            _pool.emplace(std::move(result));
            _condition.notify_one();
        }

    private:
        size_t _capacity;
        std::stack<EvaluationResult<T>> _pool;
        std::mutex _mutex;
        std::condition_variable _condition;
    };



}