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

#include <string>
#include <cstdarg>
#include <locale>
#include <vector>

#include "./libaln/aln.h"

inline std::string const& AsciiMap()
{
    static const auto map = std::string(" .,:;ox%#@");
    return map;
}

template<size_t width, size_t height, typename T, typename RandomAccessIterator>
void PrintResult(T const expected, T const actual, RandomAccessIterator const dataPoint, double offset, double range)
{
    std::cout
        << std::endl
        << expected << " => " << actual;

    auto const& map = AsciiMap();
    const auto mapSize = map.size();
        
    for (auto y = 0; y < height; y++)
    {
        std::string row;
        for (auto x = 0; x < width; x++)
        {
            auto index = y * width + x;

            double const pixelValue = (dataPoint[index] - offset) / range;
            assert(pixelValue >= 0 && pixelValue <= 1);
    
            auto const asciiValue = static_cast<size_t>(std::round(pixelValue * mapSize));
            row += map[asciiValue];
        }
        std::cout << row << std::endl;
    }
}

template<typename T>
size_t CountClassificationErrors(
    aln::Network<T> const& aln, 
    aln::DataPoint<T> const& expected,
    aln::DataSet<T> const& inputs)
{
    std::atomic<size_t> errors{ 0 };

    aln::EvaluateMany(
        aln::Parallel,
        aln,
        cbegin(inputs),
        inputs.size(),
        [&](auto const& _, auto const index, auto const& point, auto const result)
        {
            // a classification error if result is zero, 
            // or if sign of result does not match sign of expected value
            if (result == 0 || std::signbit(result) != std::signbit(expected[index]))
                errors++;
        });

    return errors;
}

template<typename T, typename TLabel>
aln::DataPoint<T> SubstituteClassLabels(
    aln::DataPoint<T> const& classLabels, 
    TLabel const label, 
    TLabel const substitutionValue = 1)
{
    aln::DataPoint<T> substitutions(classLabels.size());

    auto const range = aln::make_range(classLabels.size());
    std::for_each_n(
        aln::ParallelUnsequenced,
        cbegin(range),
        range.size(),
        [&](auto const index)
        {
            auto const distanceFromTarget = classLabels[index] - static_cast<T>(label);
            auto const substitution = std::fabs(distanceFromTarget) < 0.5
                ? substitutionValue
                : -substitutionValue;

            substitutions[index] = static_cast<T>(substitution);
        });

    return substitutions;
}

template<typename T, typename TLabel = T>
aln::Network<T> BuildBinaryClassificationNetwork(
    size_t const dimensions,
    double desiredRMSE,
    TLabel const labelOutputValue = 1)
{
    auto aln = aln::Network<T>(dimensions);

    // set up a min of max tree to constrain classification output to be in [-outputBound, outputBound]
    auto minIndex = aln.Split(aln.RootIndex(), aln::MinMaxType::Min);
    auto leftMinChildIndex = aln.MinMaxNodeAt(minIndex).LeftIndex();
    auto rightMinChildIndex = aln.MinMaxNodeAt(minIndex).RightIndex();

    // split left child of min into max
    auto maxIndex = aln.Split(leftMinChildIndex, aln::MinMaxType::Max);
    auto rightMaxChildIndex = aln.MinMaxNodeAt(maxIndex).RightIndex();

    // constrain network output to be within [-outputBound, outputBound]:
    //   - right child of min is constant to cutoff any values above outputBound
    //   - right child of max is constant to cutoff any values below -outputBound
    auto const outputBound = static_cast<T>(labelOutputValue - desiredRMSE);
    {
        // scope: references only valid until next call to split

        auto& minCutoff = aln.LinearUnitAt(rightMinChildIndex);
        minCutoff.SetConstantValue(outputBound);

        auto& maxCutoff = aln.LinearUnitAt(rightMaxChildIndex);
        maxCutoff.SetConstantValue(-outputBound);
    }

    return aln;
}

template <typename T> 
aln::DataSet<T> MakeIntensityNormalization(size_t dimensions, size_t outputIndex, T maxIntensity = 255)
{
    aln::DataSet<T> dataSet(2);
    
    dataSet[0].resize(dimensions);
    std::fill_n(begin(dataSet[0]), dimensions, static_cast<T>(0));

    dataSet[1].resize(dimensions);
    std::fill_n(begin(dataSet[1]), dimensions, maxIntensity);

    if (outputIndex < dimensions)
    {
        // don't normalize output
        dataSet[0][outputIndex] = 0;
        dataSet[1][outputIndex] = 1;
    }

    return dataSet;
}

template <typename T>
std::tuple<aln::DataPoint<T>, aln::DataSet<T>> SliceColumn(aln::DataSet<T> const& dataSet, size_t columnToSlice)
{
    if (dataSet.size() == 0)
        return {};

    aln::DataPoint<T> slice(dataSet.size());    // a single vector of the sliced column
    aln::DataSet<T> remainder(dataSet.size());  // a data set containing all the other columns

    auto const range = aln::make_range(dataSet.size());
    std::for_each_n(
        aln::ParallelUnsequenced,
        cbegin(range),
        range.size(),
        [&](auto const index)
        {
            auto const& sourceRow = dataSet[index];
            auto& remainderRow = remainder[index];
            remainderRow.resize(sourceRow.size() - 1);

            slice[index] = sourceRow[columnToSlice];
            
            auto const beforeCount = columnToSlice;
            auto const afterCount = remainderRow.size() - beforeCount;
            if (beforeCount > 0)
            {
                // copy from beginning of row up to the sliced column
                std::copy_n(cbegin(sourceRow), beforeCount, begin(remainderRow));
            }
            if (afterCount > 0)
            {
                // copy everything after the sliced column to end of row
                std::copy_n(cbegin(sourceRow) + beforeCount + 1, afterCount, begin(remainderRow) + beforeCount);
            }
        });

    return std::tie(slice, remainder);
}
