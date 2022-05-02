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

#include <mutex>
#include "aln.h"

namespace aln
{
    enum class SplitPreference
    {
        DataDriven = 0,
        Min = 1,
        Max = 2
    };


    /// <summary>
    /// Split a network based on the error of the network on a data set.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="desiredRMSError">The desired RMS error to achieve on the data set.</param>
    /// <param name="minPointsPerSplit">The minimum number of points linear unit must be responsible for before it can be split.</param>
    /// <param name="maxSplits">The maximum number of splits allowed.</param>
    /// <param name="expectedValues">Iterator to the first point of the expected values corresponding to the sequence of points in the data set.</param>
    /// <param name="dataSet">Iterator to the first point of the data set.</param>
    /// <param name="count">The number of data points to evaluate.</param>
    /// <param name="resultPool">A pool of results that can be used during evaluation. The capacity of the pool may limit parallelism.</param>
    /// <param name="splitPreference">Split preference.</param>
    /// <returns>The number of splits performed.</returns>
    template <typename T, typename ExpectedValueRandomAccessIterator, typename DataSetRandomAccessIterator>
    size_t Split(
        Network<T>& aln, 
        double const desiredRMSError, 
        size_t const minPointsPerSplit,
        size_t const maxSplits,
        ExpectedValueRandomAccessIterator const expectedValues,
        DataSetRandomAccessIterator const dataSet,
        size_t const count,
        EvaluationResultPool<T>& resultPool,
        SplitPreference const splitPreference = SplitPreference::DataDriven)
    {
        if (maxSplits == 0)
            return 0;

        if (minPointsPerSplit < 2)
            throw std::invalid_argument("min points per split must be at least 2");

        auto const dimensions = aln.Dimensions();

        struct SplitStats
        {
            SplitStats()
                : count{}
                , sse{}
            {
            }

            size_t count;           // number of points seen
            double sse;             // total squared error accumulated during adaptation
            std::vector<double> b;  // average convexity

            std::mutex mutex;       // prevent simultaneous updates
        };

        // calculate linear unit split stats
        auto& linearUnits = aln.LinearUnits();
        std::vector<SplitStats> splitStats(linearUnits.size());

        EvaluateMany(
            Parallel,
            aln,
            dataSet,
            count,
            resultPool,
            [&](auto const&, auto const index, auto const& dataPoint, EvaluationResult<T> const& result)
            {
                // update split stats for the responsible linear unit
                auto const error = static_cast<double>(result.Value()) - static_cast<double>(expectedValues[index]);
                auto const luIndex = result.SurfaceResponsibility();
                auto const& lu = aln.LinearUnitAt(luIndex);
                
                // lock and update stats for this LTU
                auto& stats = splitStats[luIndex.Ordinal()];
                std::unique_lock<std::mutex> lock(stats.mutex);

                if (stats.b.size() != dimensions)
                    stats.b.resize(dimensions);

                stats.count++;
                stats.sse += error * error;

                // update convexity measure
                auto const& c = lu.C();
                auto const& d = lu.D();

                auto const range = make_range(dimensions);
                std::for_each_n(
                    Sequential,
                    cbegin(range),
                    range.size(),
                    [&](auto i)
                    {
                        // from NANO AdaptLFN():
                        // We analyze the errors of sample value minus ALN value V - L = -fltError (N.B. minus) on the piece which are
                        // further from and closer to the centroid than the stdev of the points on the piece along the current axis.
                        // If the V - L  is positive (negative) away from the center compared to the error closer to the center,
                        // then we need a split of the LFN into a MAX (MIN) node.

                            auto x = dataPoint[i];

                            // d[i] is variance of points on the piece along axis i
                            // x - c[i] * x - c[i] is squared distance from current point to centroid
                            auto xmc = x - c[i];
                            auto diff = (xmc * xmc) - d[i];
                            auto bend = diff > 0 ? -diff * error : diff * error;

                            stats.b[i] += bend;
                    });
            });

        // visit all leaf nodes and determine whether to split
        std::mutex splitMutex;
        std::vector<std::tuple<NodeIndex, double, double>> toSplit;
        auto const range = make_range(linearUnits.size());
        std::for_each_n(
            Sequential,
            cbegin(range),
            range.size(),
            [&](auto i)
            {
                auto& lu = linearUnits[i];
                auto& stats = splitStats[i];
                if (stats.count == 0 || !lu.IsSplitAllowed())
                    return;

                // TODO: see NANO doSplits() for split count thresholds
                // there seems to be different thresholds for classification and regression;
                // ... we'll use the classification path for now

                auto const luRMSError = std::sqrt(stats.sse / static_cast<double>(stats.count));
                if (stats.count >= minPointsPerSplit && luRMSError > desiredRMSError)
                {
                    // calculate overall convexity for the linear unit
                    auto t = std::reduce(
                        ParallelUnsequenced,
                        begin(stats.b),
                        end(stats.b));

                    std::unique_lock<std::mutex> lock(splitMutex);
                    toSplit.emplace_back(aln.IndexOf(lu), t, luRMSError);
                }
            });

        // if we have a limited number of splits, 
        // sort by count in descending order so that we split the ones with most responsibility first
        auto const splitCount = (std::min)(toSplit.size(), maxSplits);
        if (toSplit.size() > splitCount)
        {
            std::sort(
                begin(toSplit),
                end(toSplit),
                [](auto const& first, auto const& second)
                {
                    auto const firstRmse = std::get<2>(first);
                    auto const secondRmse = std::get<2>(second);
                    return firstRmse > secondRmse; // strict weak ordering
                });
        }

        for (size_t i = 0U; i < splitCount; i++)
        {
            NodeIndex splitIndex;
            double t;
            double rmse;
            std::tie(splitIndex, t, rmse) = toSplit[i];

            MinMaxType nodeType;
            if (splitPreference == SplitPreference::Min)
            {
                nodeType = MinMaxType::Min;
            }
            else if (splitPreference == SplitPreference::Max)
            {
                nodeType = MinMaxType::Max;
            }
            else
            {
                if (t == 0) // is there some kind of epsilon we can put on this???
                {
                    // use same type as parent
                    auto parentIndex = aln.LinearUnitAt(splitIndex).ParentIndex();
                    nodeType = parentIndex.IsMinMax()
                        ? aln.MinMaxNodeAt(parentIndex).Type()
                        : MinMaxType::Min; // parent is not a min/max node, choose min type???
                }
                else if (t > 0)
                {
                    // value of samples are higher than the linear unit surface some distance from centroid
                    nodeType = MinMaxType::Max;
                }
                else
                {
                    nodeType = MinMaxType::Min;
                }
            }

            aln.Split(splitIndex, nodeType);
        }

        return splitCount;
    }
}