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

#include <functional>
#include <memory>
#include <type_traits>

#include "aln.h"

namespace aln
{
    namespace details
    {
        /// <summary>
        /// Evaluates the linear unit at the specified ordinal position in the network.
        /// </summary>
        template<typename T, typename RandomAccessIterator>
        T EvaluateLinearUnit(
            Network<T> const& aln,
            size_t const ordinal,
            RandomAccessIterator const point)
        {
            auto const dimensions = aln.Dimensions();
            auto const& lu = aln.LinearUnitAt(ordinal);
            auto const& w = lu.W();
            auto const bias = lu.Bias();
            
            // multiply/add weights with input values; sum initialized to bias value
            return std::transform_reduce(
                Sequential, // how many points before Parallel becomes viable?
                point,
                point + dimensions,
                cbegin(w),
                bias,
                std::plus<>{},   
                std::multiplies<>{});
        }

        template<typename T>
        struct LazyEvaluation
        {
            constexpr static T Maximum = std::numeric_limits<T>::max();
            constexpr static T Minimum = -Maximum;

            std::optional<T> KnownMin;
            std::optional<T> KnownMax;

            bool CanShortCircuit(T value, MinMaxType minOrMax)
            {
                if (minOrMax == MinMaxType::Max)
                {
                    // can short circuit if value is greater than or equal to previously seen minimum
                    if (KnownMin && value >= *KnownMin)
                        return true;

                    KnownMax = (std::max)(value, KnownMax.value_or(Minimum));
                }
                else
                {
                    // can short circuit if value is less than or equal to previously seen maximum
                    if (KnownMax && value <= *KnownMax)
                        return true;

                    KnownMin = (std::min)(value, KnownMin.value_or(Maximum));
                }

                return false;
            }
        };

        /// <summary>
        /// Evaluates the specified node and returns the responsible linear unit.
        /// Assumes all linear units have already been evaluated with values
        /// stored in <paramref name="result"/>
        /// </summary>
        /// <param name="aln"></param>
        /// <param name="nodeIndex"></param>
        /// <param name="point">Iterator to the first element of the input point.</param>
        /// <param name="lazyEvaluation">Holds information that can be used for short-circuiting evaluation of child nodes.</param>
        /// <param name="result">A pointer to an instance that on return, will contain the result of evaluation of the network on the input point.</param>
        /// <param name="previousRoute">A pointer to the previous route used for the input point; or <c>nullptr</c> is there isn't one.</param>
        /// <returns>A tuple containing the index and the value of the linear unit responsible for computing the value of the surface at this node.</returns>
        template<typename T, typename RandomAccessIterator>
        std::tuple<NodeIndex, T> EvaluateNode(
            Network<T> const& aln,
            NodeIndex const nodeIndex,
            RandomAccessIterator const point,
            LazyEvaluation<T> lazyEvaluation, // note: always passed by value
            EvaluationResult<T>* const result,
            EvaluationRoute const* const previousRoute)
        {
            auto const ordinal = nodeIndex.Ordinal();

            if (nodeIndex.IsLinear())
            {
                auto value = result
                    ? result->ValueOf(nodeIndex)
                    : NaN<T>();

                if (!IsComputed(value))
                {
                    value = EvaluateLinearUnit(aln, ordinal, point);
                    assert(IsComputed(value));

                    if (result)
                        result->SetLinearResult(nodeIndex, value);
                }
                return std::tie(nodeIndex, value);
            }

            auto const& minMaxNode = aln.MinMaxNodeAt(ordinal);
            auto const leftIndex = minMaxNode.LeftIndex();
            auto const rightIndex = minMaxNode.RightIndex();

            struct ChildResult
            {
                NodeIndex Index;
                std::tuple<NodeIndex, T> SurfaceResult; 

                NodeIndex& SurfaceResponsibility() { return std::get<0>(SurfaceResult); }
                T& Value() { return std::get<1>(SurfaceResult); }
            };
            std::array<ChildResult, 2> childResults;

            // is there a known responsible child from previous evaluation of this point?
            auto firstIndex = NodeIndex::Invalid();
            if (previousRoute)
            {
                if (!previousRoute->IsEmpty())
                {
                    firstIndex = previousRoute->GetResponsibleChildOf(nodeIndex).value_or(NodeIndex::Invalid());
                }
            }

            if (firstIndex.IsInvalid())
            {
                childResults[0].Index = leftIndex;
                childResults[1].Index = rightIndex;
            }
            else 
            {
                assert(firstIndex == leftIndex || firstIndex == rightIndex);
                childResults[0].Index = firstIndex == leftIndex ? leftIndex : rightIndex;
                childResults[1].Index = firstIndex == leftIndex ? rightIndex : leftIndex;
            }

            // evaluate first child
            childResults[0].SurfaceResult = EvaluateNode(
                aln, 
                childResults[0].Index, 
                point, 
                lazyEvaluation,
                result, 
                previousRoute);
            
            // it is responsible if we can short circuit based on this value
            auto firstIsResponsible = lazyEvaluation.CanShortCircuit(childResults[0].Value(), minMaxNode.Type());
            if (!firstIsResponsible)
            {
                // otherwise we have to evaluate second child too...
                childResults[1].SurfaceResult = EvaluateNode(
                    aln, 
                    childResults[1].Index, 
                    point, 
                    lazyEvaluation,
                    result, 
                    previousRoute);

                // ... and then calculate which child is responsible
                auto const firstValue = childResults[0].Value();
                auto const secondValue = childResults[1].Value();
                firstIsResponsible =
                    (minMaxNode.IsMin() && firstValue < secondValue) ||
                    (minMaxNode.IsMax() && firstValue > secondValue);
            }
               
            auto& nodeResult = firstIsResponsible ? childResults[0] : childResults[1];

            // min max result is the index of the responsible child and its value
            if (result)
                result->SetMinMaxResult(nodeIndex, std::tie(nodeResult.Index, nodeResult.Value()));

            return nodeResult.SurfaceResult;
        }

        /// <summary>
        /// Performs an evaluation of the network on the specified input vector.
        /// </summary>
        /// <param name="aln"></param>
        /// <param name="point">Iterator to the first element of the input point.</param>
        /// <param name="result">A pointer to an instance that on return, will contain the result of evaluation of the network on the input point.</param>
        /// <param name="previousRoute">A pointer to the previous route used for the input point; or <c>nullptr</c> is there isn't one.</param>
        /// <remarks>
        /// <para>
        /// This method may be safely called in parallel with other calls to <see cref="Evaluate()"/> as long as each parallel invocation
        /// is using a different instance of an <see cref="EvaluationResult"/>.
        /// </para>
        /// <para>
        /// The random access iterator <paramref name="first"/> must be capable of iterating over <see cref="Dimensions()"/> values.
        /// </para>
        /// </remarks>
        template <typename T, typename RandomAccessIterator>
        T Evaluate(
            Network<T> const& aln,
            RandomAccessIterator const point,
            EvaluationResult<T>* const result = nullptr,
            EvaluationRoute const* const previousRoute = nullptr)
        {
            if (result)
                result->Reset(aln);

            NodeIndex surfaceResponsibility;
            T value;

            std::tie(surfaceResponsibility, value) = details::EvaluateNode(
                aln,
                aln.RootIndex(),
                point,
                {},
                result,
                previousRoute);

            assert(surfaceResponsibility.IsLinear());

            if (result)
            {
                assert(value == result->ValueOf(surfaceResponsibility));
                result->SetSurfaceResponsibility(surfaceResponsibility);
                result->SetValue(value);
            }

            return value;
        }
    }

    

    /// <summary>
    /// Performs an evaluation of the network on the specified input vector.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="point">Iterator to the first element of the input point.</param>
    /// <param name="result">An optional reference that on return, will contain the result of evaluation of the network on the input point.</param>
    /// <param name="previousRoute">An optional reference to the previous route used for the input point.</param>
    /// <remarks>
    /// <para>
    /// This method may be safely called in parallel with other calls to <see cref="Evaluate()"/> as long as each parallel invocation
    /// is using a different instance of an <see cref="EvaluationResult"/>.
    /// </para>
    /// <para>
    /// The random access iterator <paramref name="first"/> must be capable of iterating over <see cref="Dimensions()"/> values.
    /// </para>
    /// </remarks>
    template <typename T, typename RandomAccessIterator>
    T Evaluate(
        Network<T> const& aln,
        RandomAccessIterator const point,
        EvaluationResult<T>& result)
    {
        return details::Evaluate(aln, point, &result);
    }

    /// <summary>
    /// Performs an evaluation of the network on the specified input vector.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="point">Iterator to the first element of the input point.</param>
    /// <returns>The result of evaluation of the network on the input point.</returns>
    /// <remarks>
    /// <para>
    /// This method may be safely called in parallel with other calls to <see cref="Evaluate()"/> as long as each parallel invocation
    /// is using a different instance of an <see cref="EvaluationResult"/>.
    /// </para>
    /// <para>
    /// The random access iterator <paramref name="first"/> must be capable of iterating over <see cref="Dimensions()"/> values.
    /// </para>
    /// </remarks>
    template <typename T, typename RandomAccessIterator>
    [[nodiscard]] 
    T Evaluate(
        Network<T> const& aln,
        RandomAccessIterator const point)
    {
        return details::Evaluate(aln, point);
    }

    
    /// <summary>
    /// Evaluate the network on many data points.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="dataSet">Iterator to the first point of the data set.</param>
    /// <param name="count">The number of data points to evaluate.</param>
    /// <param name="resultPool">A pool of results that can be used during evaluation. The capacity of the pool may limit the number of parallel invocations of <paramref name="callback"/>.</param>
    /// <param name="callback">The callback to be invoked for each evaluation.</param>
    /// <remarks>
    /// Caution using std::execution::par_unseq if callback may acquire locks or otherwise may not be interleaved on the same thread.
    /// </remarks>
    template<typename ExecutionPolicy, typename T, typename RandomAccessIterator, typename Callback>
    void EvaluateMany(
        ExecutionPolicy executionPolicy,
        Network<T> const& aln,
        RandomAccessIterator const dataSet,
        size_t const count,
        EvaluationResultPool<T>& resultPool,
        Callback callback)
    {
        // Callback should be function type compatible with 
        //    void(Network<T> const& aln, size_t index, decltype(*std::declval{RandomAccessIterator const}()) dataSet, EvaluationResult<T> const& result)

        auto const range = make_range(count);
        std::for_each_n(
            executionPolicy, 
            cbegin(range),
            range.size(),
            [&](auto const index)
            {
                auto const& dataPoint = dataSet[index];
                auto result = resultPool.Acquire();
                try
                {
                    auto const firstElement = cbegin(dataPoint);

                    Evaluate(
                        aln, 
                        firstElement,
                        result);

                    callback(aln, index, dataPoint, result);

                    resultPool.Release(std::move(result));
                }
                catch (...)
                {
                    resultPool.Release(std::move(result));
                    throw;
                }
            });
    }

    /// <summary>
    /// Evaluate the network on many data points.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="dataSet">Iterator to the first point of the data set.</param>
    /// <param name="count">The number of data points to evaluate.</param>
    /// <param name="callback">The callback to be invoked for each evaluation.</param>
    /// <remarks>
    /// Caution using std::execution::par_unseq if callback may acquire locks or otherwise may not be interleaved on the same thread.
    /// </remarks>
    template<typename ExecutionPolicy, typename T, typename RandomAccessIterator, typename Callback>
    void EvaluateMany(
        ExecutionPolicy executionPolicy,
        Network<T> const& aln,
        RandomAccessIterator const dataSet,
        size_t const count,
        Callback callback)
    {
        // Callback should be function type compatible with 
        //    void(Network<T> const& aln, size_t index, decltype(*std::declval{RandomAccessIterator const}()) dataSet, T const result)

        auto const range = make_range(count);
        std::for_each_n(
            executionPolicy,
            cbegin(range),
            range.size(),
            [&](auto const index)
            {
                auto const& dataPoint = dataSet[index];
                auto const firstElement = cbegin(dataPoint);
                auto result = Evaluate(aln, firstElement);
                callback(aln, index, dataPoint, result);
            });
    }

    /// <summary>
    /// Calculate the root mean squared sum of errors of the network on the specified data set.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="expectedValues">Iterator to the first point of the expected values corresponding to the sequence of points in the data set.</param>
    /// <param name="dataSet">Iterator to the first point of the data set.</param>
    /// <param name="count">The number of data points to evaluate.</param>
    /// <returns></returns>
    /// <remarks>
    /// Assumes that the output dimension element of each point in the data set contains the desired 
    /// value of the network surface on the domain defined by the other element values of the point.
    /// </remarks>
    template<typename T, typename ExpectedValueRandomAccessIterator, typename DataSetRandomAccessIterator>
    double CalculateRMSError(
        Network<T> const& aln,
        ExpectedValueRandomAccessIterator const expectedValues,
        DataSetRandomAccessIterator const dataSet,
        size_t const count)
    {
        std::atomic<double> sse{ 0 };
        std::atomic<size_t> total{ 0 };

        EvaluateMany(
            Parallel,
            aln,
            dataSet,
            count,
            [&](auto const& _, auto const index, auto const& dataPoint, auto const result)
            {
                auto const expectedValue = static_cast<double>(expectedValues[index]);
                auto const error = static_cast<double>(result) - expectedValue;
                auto const squaredError = error * error;

                total++;

                // update sse 
                // ... atomic<double>::operator+= not currently defined, so use compare_exchange
                auto current = sse.load();
                double desired;
                do
                {
                    desired = current + squaredError;
                } while (!sse.compare_exchange_weak(current, desired));
            });

        return std::sqrt(sse / static_cast<double>(total));
    }
}