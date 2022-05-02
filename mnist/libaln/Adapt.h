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

#include "aln.h"

namespace aln
{
    /// <summary>
    /// Adapts the network to reduce the distance between the network surface and the value at <see cref="OutputIndex()"/> 
    /// represented by the specified evaluation result, which should be obtained by calling <see cref="Evaluate()"/> prior
    /// to calling <see cref="Adapt()"/>.
    /// </summary>
    /// <param name="aln"></param>
    /// <param name="adaptRate">Adaptation rate in (0, 1].</param>
    /// <param name="desiredRMSError">The desired RMS error to achieve on the data set.</param>
    /// <param name="point">Iterator to the first element of the input point.</param>
    /// <param name="evalResult"></param>
    /// <param name="positiveClassRatio">For regression data, leave at 1.0. For binary classification domains, this is the expected proportion of positive class samples to negative class samples.</param>
    /// <returns>True if adaptation occurred; otherwise <see cref="false"/>, for example, if the responsible <see cref="LinearUnit"/> is constant.</returns>
    /// <remarks>
    /// <para>
    /// The random access iterator <paramref name="point"/> must be capable of iterating over as many values as there are dimensions in the network.
    /// </para>
    /// </remarks>
    template<typename T, typename RandomAccessIterator>
    bool Adapt(
        Network<T>& aln, 
        double const adaptRate, 
        double const desiredRMSError,
        T const expectedValue,
        RandomAccessIterator const point,
        EvaluationResult<T> const& evalResult,
        double const positiveClassRatio = 1.0)
    {
        if (adaptRate <= 0 || adaptRate > 1)
            throw std::invalid_argument("invalid adaptation rate");

        if (!IsComputed(evalResult.Value()))
            throw std::invalid_argument("invalid result value");

        if (evalResult.SurfaceResponsibility().IsInvalid())
            throw std::invalid_argument("invalid result surface responsibility");

        // currently we only adapt the responsible linear unit;
        // if we start using smoothing fillets we may need to track proportional responsibility

        auto const error = static_cast<double>(evalResult.Value()) - static_cast<double>(expectedValue);

        auto surfaceResponsibility = evalResult.SurfaceResponsibility();
        auto& lu = aln.LinearUnitAt(surfaceResponsibility);
        if (lu.IsConstant())
            return false;

        auto const dimensions = aln.Dimensions();
        auto const& constraints = aln.Constraints();

        auto& w = lu.W();
        auto& c = lu.C();
        auto& d = lu.D();

        // the following algorithm from AdaptLFN in the NANO source on GitHub

        // learning boost for classification problems
        auto const classLearningBoost = std::fabs(error) > desiredRMSError
            ? positiveClassRatio
            : 1.0;

        // we correct error by adapting weights and centroid; so distribute proportion
        // of adaptation rate equally between the of values in centroid and weights that will be adapted
        auto const distributedRate = classLearningBoost * adaptRate / static_cast<double>((c.size() + w.size()));
       
        // shared update lambda
        auto updateFn = [&](auto i)
        {
            // distance from point to pre-adapted centroid (L - V)
            auto const xMinusC = i < dimensions
                ? point[i] - c[i]
                : expectedValue - c[i];

            if (i < dimensions)
            {
                auto& constraint = constraints[i];
                if (constraint.MaxWeight() > constraint.MinWeight())
                {
                    // update variance before updating centroid
                    d[i] += static_cast<T>((xMinusC * xMinusC - d[i]) * adaptRate); // this rate not used for error correction
                    d[i] = std::max(d[i], static_cast<T>(0.0000001));               // d cannot be zero

                    // update weight preserving units: output unit / input unit of this axis
                    w[i] -= static_cast<T>(error * distributedRate * xMinusC / d[i]);
                    w[i] = std::clamp(w[i], constraint.MinWeight(), constraint.MaxWeight());

                }
            }

            // update centroid using exponential smoothing
            c[i] += static_cast<T>(xMinusC * distributedRate);
        };

        // do all weight/centroid/variance updates
        auto const range = make_range(dimensions + 1);
        std::for_each_n(
            Sequential,
            cbegin(range),
            range.size(),
            std::move(updateFn));

        //
        // compress weighted centroid into bias element:
        // bias = sum(-w[i] * c[i])       
        //
        
        struct bias_multiplies
        {
            constexpr T operator()(const T& w, const T& c) const 
            {
                return -w * c;
            }
        };

        auto const bias = std::transform_reduce(
            Sequential,
            cbegin(w),
            cend(w),
            cbegin(c),
            c[dimensions], // implicit -1 output weight on centroid
            std::plus<>{},
            bias_multiplies{});

        lu.SetBias(bias);

        return true;
    }
}