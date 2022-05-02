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

#include <algorithm>
#include <exception>
#include <cmath>
#include <vector>
#include <random>

#include "aln.h"

namespace aln
{
    /// <summary>
    /// Trains a network on a data set for a specified number of epochs or until the minimum RMS error is achieved.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="epochs"></param>
    /// <param name="adaptRate">Adaptation rate in (0, 1].</param>
    /// <param name="desiredRMSError">The desired RMS error to achieve on the data set.</param>
    /// <param name="expectedValues">Iterator to the first point of the expected values corresponding to the sequence of points in the data set.</param>
    /// <param name="dataSet">Iterator to the first point of the data set.</param>
    /// <param name="count">The number of data points to evaluate.</param>
    /// <param name="totalEvaluations">The total number of node evaluations performed during training.</param>
    /// <param name="resultPool">A pool of results that can be used during evaluation. The capacity of the pool may limit the number of parallel invocations of <paramref name="callback"/>.</param>
    /// <param name="rng"></param>
    /// <param name="positiveClassRatio">For regression data, leave at 1.0. For binary classification domains, this is the expected proportion of positive class samples to negative class samples.</param>
    /// <returns>The RMS error achieved on the dataset by the network.</returns>
    template<typename T, typename ExpectedValueRandomAccessIterator, typename DataSetRandomAccessIterator, typename UniformRandomNumberGenerator>
    double Train(
        Network<T>& aln, 
        size_t const epochs,
        double const adaptRate,
        double const desiredRMSError,
        ExpectedValueRandomAccessIterator const expectedValues,
        DataSetRandomAccessIterator const dataSet,
        size_t const count,
        EvaluationResultPool<T>& resultPool,
        size_t& totalEvaluations,
        UniformRandomNumberGenerator& rng,
        double positiveClassRatio = 1.0)
    {
        if (epochs == 0)
            throw std::invalid_argument("invalid epochs");

        if (adaptRate <= 0 || adaptRate > 1)
            throw std::invalid_argument("invalid adaptRate");

        if (desiredRMSError < 0)
            throw std::invalid_argument("invalid desiredRMSError");

        if (count == 0)
            throw std::invalid_argument("invalid count");

        totalEvaluations = 0;

        // we init a vector of row indexes...
        // this vector can be shuffled at the start of each epoch
        std::vector<size_t> shuffled(count);
        auto const range = make_range(count);
        std::for_each_n(
            ParallelUnsequenced,
            cbegin(range),
            range.size(),
            [&](auto const index)
            {
                shuffled[index] = index;
            });

        // routes for each point to speed up eval
        std::vector<EvaluationRoute> routes(count);

        // track overall rmse achieved during each epoch of training
        double rmse = NaN<double>();
        auto needActualRmse = true;

        for (size_t epoch = 0; epoch < epochs; epoch++)
        {
            // shuffle the data points
            std::shuffle(begin(shuffled), end(shuffled), rng);

            // evaluate and adapt to each point
            double sse = 0;
            auto evalResult = resultPool.Acquire(); // we can re-use a single result for each iteration
            try
            {
                for (size_t i = 0; i < count; i++)
                {
                    size_t pointIndex = shuffled[i];

                    auto const expectedValue = expectedValues[pointIndex];
                    auto const dataPoint = cbegin(dataSet[pointIndex]);
                    auto& evalRoute = routes[pointIndex];
                    
                    // evaluate with the last known route for the point
                    details::Evaluate(aln, dataPoint, &evalResult, &evalRoute);
                    auto completedEvaluations =  evalResult.TotalEvaluations();
                    assert(completedEvaluations > 0);
                    totalEvaluations += completedEvaluations;

                    // update route based on result
                    evalRoute.Update(aln.RootIndex(), evalResult);
                    
                    auto const error = static_cast<double>(evalResult.Value()) - static_cast<double>(expectedValue);
                    sse += error * error;

                    Adapt(aln, adaptRate, desiredRMSError, expectedValue, dataPoint, evalResult, positiveClassRatio);
                }

                resultPool.Release(std::move(evalResult));
            }
            catch (...)
            {
                resultPool.Release(std::move(evalResult));
                throw;
            }

            // update rmse estimate after epoch
            rmse = std::sqrt(sse / static_cast<double>(count));
            needActualRmse = true;

            // see if we can end training early
            if (rmse <= desiredRMSError)
            {
                // rmse is just an estimate since it doesn't account for
                // adaptations after each point; 
                
                // now that we think its low enough, calculate for real...

                rmse = CalculateRMSError(aln, expectedValues, dataSet, count);
                needActualRmse = false;

                if (rmse <= desiredRMSError)
                    break; // break out of epoch
            }

        }

        if (needActualRmse)
            rmse = CalculateRMSError(aln, expectedValues, dataSet, count);

        return rmse;
    }


}