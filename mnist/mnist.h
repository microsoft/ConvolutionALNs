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
#include <cassert>
#include <chrono>
#include <execution>
#include <iostream>
#include <random>
#include <sstream>

// uncomment below to force all C++ standard parallel algorithms to run sequentially
// #define ALN_DEBUG_FORCE_SEQUENTIAL

#include "./libaln/aln.h"
#include "utils.h"

namespace mnist
{
    template <typename T>
    class DigitEvaluator
    {
        static_assert(std::is_floating_point_v<T>);

    public:
        DigitEvaluator(std::string const& alnFolderPath)
        {
            std::for_each_n(
                aln::ParallelUnsequenced,
                begin(_digitRange),
                _digitRange.size(),
                [&](auto const digit)
                {
                    std::stringstream pathStream;
                    pathStream << alnFolderPath << "/aln-digit-" << digit << ".json";

                    auto aln = aln::JsonImport<T>(pathStream.str());

                    _alns[digit] = std::move(aln);
                });
        }

        template<typename RandomAccessIterator>
        int Evaluate(RandomAccessIterator const dataPoint) const
        {
            // evaluate the ten classifiers in parallel
            std::array<T, 10> values;
        
            std::for_each_n(
                aln::ParallelUnsequenced,
                begin(_digitRange),
                _digitRange.size(),
                [&](auto const digit)
                {
                    auto const& aln = _alns[digit];
                    values[digit] = aln::Evaluate(aln, dataPoint);
                });


            // the index of the (first) classifier with the max value is the resulting label
            auto maxElement = std::max_element(begin(values), end(values));
            auto maxIndex = maxElement - begin(values);

            return static_cast<int>(maxIndex);
        }

    private:
        std::array<aln::Network<T>, 10> _alns;
        inline static auto const _digitRange = aln::make_range(10);
    };

    template <typename ExecutionPolicy, typename T>
    std::vector<int> Evaluate(
        ExecutionPolicy&& executionPolicy,
        DigitEvaluator<T>& evaluator, 
        aln::DataSet<T> const& dataSet)
    {
        // expecting data set rows to be 784 elements each; 28x28 matrix in row major order
        if (dataSet.size() > 0 && dataSet[0].size() != 784)
            throw std::invalid_argument("invalid data set dimension");

        // gather the label results 
        std::vector<int> labelResults(dataSet.size());
        auto const indexRange = aln::make_range(dataSet.size());
        std::for_each_n(
            executionPolicy,
            begin(indexRange),
            indexRange.size(),
            [&](auto const pointIndex)
            {
                auto const& point = dataSet[pointIndex];
                labelResults[pointIndex] = evaluator.Evaluate(cbegin(point));
            });

        return labelResults;
    }

    template <typename T>
    void TrainDigit(
        aln::Network<T>& aln,
        std::string const& alnPath,
        int const digitToClassify, 
        double desiredRmse,
        size_t const iterations, 
        aln::DataPoint<T> const& trainLabels,
        aln::DataSet<T> const& trainInput,
        aln::DataPoint<T> const& testLabels,
        aln::DataSet<T> const& testInput,
        aln::DataSet<T> const& normalization)
    {
        static_assert(std::is_floating_point_v<T>);

        //  outputs => desired class/label in [0, 9]
        //  remaining 784 elements => 28x28 matrix in row major order, elements in [0, 255]
        auto const dimensions = 784;

        if (trainInput.size() == 0)
            throw std::invalid_argument("empty training set");

        if (testInput.size() == 0)
            throw std::invalid_argument("empty test set");

        if (trainInput.size() != trainLabels.size())
            throw std::invalid_argument("mismatched training data input/output row counts");

        if (testInput.size() != testLabels.size())
            throw std::invalid_argument("mismatched testing data input/output row counts");

        // expecting data set rows to be 784 elements each
        if (trainInput[0].size() != 784)
            throw std::invalid_argument("invalid training data dimensions");

        if (testInput[0].size() != 784)
            throw std::invalid_argument("invalid testing data dimensions");

        // replace label values with value 1 if it matches target class, -1 otherwise
        auto const trainExpected = SubstituteClassLabels(trainLabels, digitToClassify);
        auto const digitsInTrainSet = std::count_if(
            aln::ParallelUnsequenced,
            cbegin(trainExpected),
            cend(trainExpected),
            [](auto const value)
            {
                return value == static_cast<T>(1);
            });

        auto const testExpected = SubstituteClassLabels(testLabels, digitToClassify);
        auto const digitsInTestSet = std::count_if(
            aln::ParallelUnsequenced,
            cbegin(testExpected),
            cend(testExpected),
            [](auto const value)
            {
                return value == static_cast<T>(1);
            });

        std::cout << "[" << digitToClassify << "] appears " << digitsInTrainSet << " times in the training set" << std::endl;
        std::cout << "[" << digitToClassify << "] appears " << digitsInTestSet << " times in the test set" << std::endl;
        
        // set up initial constraints and structure;
        auto const epsilon = static_cast<T>(0.1);
        auto const weightBoundMax = static_cast<T>(10);
        auto const weightBoundIncrement = static_cast<T>(0.0000005);
        auto weightBound = static_cast<T>(0.0005);

        auto const& weightNormalization = normalization[1];
        for (size_t i = 0; i < dimensions; i++)
        {
            auto const normalizationFactor = weightNormalization[i];
            auto const maxWeight = weightBound * normalizationFactor;
            auto const minWeight = -maxWeight;

            aln.SetMaxWeight(i, maxWeight);
            aln.SetMinWeight(i, minWeight);
            aln.SetEpsilon(i, normalizationFactor != 0 
                ? epsilon / normalizationFactor 
                : epsilon); // epsilon cannot be zero
        }

        aln::EvaluationResultPool<T> resultPool;
        std::mt19937_64 rng(42);

        auto rmse = CalculateRMSError(aln, cbegin(trainExpected), cbegin(trainInput), trainInput.size());
        auto classErrors = CountClassificationErrors(aln, trainExpected, trainInput);
        std::cout << "[" << digitToClassify << "] rmse=" << rmse << " classification errors=" << classErrors << std::endl;

        auto minTestClassErrors = std::numeric_limits<size_t>::max();
        auto minTestRmse = std::numeric_limits<double>::max();

        size_t const maxSplits = 512;
        size_t minPointsPerSplit = 75;// dimensions;      // should we allow some pieces to be under-determined until we have data augmentation??
        size_t allowedSplits = std::min(trainInput.size() / minPointsPerSplit, maxSplits);
        size_t totalSplits = 0;
        size_t splitIncrement = 1;
        size_t const splitLockoutIterations = 20;   // lock out splits for this many iterations after growth
        size_t splitLockout = 40;                   // initial split lockout quite high
        

        auto adaptRateMax = 0.1;
        auto const adaptRateMaxDecay = 0.999;
        auto const adaptRateMin = 0.01;
        auto const adaptRateIncrement = 0.01;
        auto adaptRate = adaptRateMin;                       

        auto maxEpochsPerIteration = 20;
        auto minEpochsPerIteration = 3;
        auto epochsPerIteration = maxEpochsPerIteration;
        auto const epochMinRmse = 0;                // always complete each epoch

        auto lastIterationRmse = rmse;
        auto smoothRmseDelta = rmse;
        auto smoothRmseDeltaRate = 0.1;
        auto smoothRmseDeltaSplitThreshold = 0.00005;
        auto const smoothRmseDeltaSplitThresholdDecay = 0.5;

        // the ratio of positive classes to negative classes
        //auto const positiveClassRatio = 1.0 / 9.0;
        auto const positiveClassRatio = 1.0;

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < iterations; i++)
        {
            auto iterationStart = std::chrono::high_resolution_clock::now();

            auto maxEvaluations = trainInput.size() * epochsPerIteration * (aln.LinearUnits().size() + aln.MinMaxNodes().size());
            size_t actualEvaluations = 0;
            rmse = aln::Train(
                aln,
                epochsPerIteration,
                adaptRate,
                epochMinRmse,
                cbegin(trainExpected),
                cbegin(trainInput),
                trainInput.size(),
                resultPool,
                actualEvaluations,
                rng, 
                positiveClassRatio);

            if (rmse <= desiredRmse)
                break;

            if (totalSplits < allowedSplits && splitLockout == 0)
            {
                // split decisions should be based on a validation set instead of train set???
                auto const splitCount = aln::Split(
                    aln,
                    desiredRmse,
                    minPointsPerSplit,
                    allowedSplits - totalSplits,
                    cbegin(trainExpected),
                    cbegin(trainInput),
                    trainInput.size(),
                    resultPool);

                totalSplits += splitCount;

                // make adjustments - if we grew the tree, we need to slow down learning for a bit
                if (splitCount > 0)
                {
                    epochsPerIteration = maxEpochsPerIteration;
                    adaptRate = adaptRateMin;
                    splitLockout = splitLockoutIterations;
                }
            }
            else if (splitLockout > 0)
            {
                splitLockout--;
            }

            auto luCount = aln.LinearUnits().size();

            auto iterationEnd = std::chrono::high_resolution_clock::now();
            auto iterationDuration = std::chrono::duration<double>(iterationEnd - iterationStart);
            auto overallDuration = std::chrono::duration<double>(iterationEnd - start);

            auto const rmseDelta = lastIterationRmse - rmse;
            smoothRmseDelta -= (smoothRmseDelta - rmseDelta) * smoothRmseDeltaRate;
            lastIterationRmse = rmse;

            std::cout
                << "[" << digitToClassify << "] iteration " << i
                << " rmse=" << rmse 
                << " rmse delta(smooth)=" << smoothRmseDelta
                << " epochs=" << epochsPerIteration
                << " adaptRate=" << adaptRate
                << " lazy=" << 1.0 - (static_cast<double>(actualEvaluations) / static_cast<double>(maxEvaluations))
                << " splits=" << totalSplits << "/" << allowedSplits
                << " lus=" << luCount
                << " iter time: " << iterationDuration.count() << "s"
                << " overall time: " << overallDuration.count() << "s"
                << std::endl;
                        
            if (rmse <= desiredRmse)
                break;

            if (i > 0)
            {
                if (adaptRateMax > adaptRateMin)
                {
                    adaptRateMax *= adaptRateMaxDecay;
                    adaptRateMax = std::max(adaptRateMin, adaptRateMax);
                }

                if (adaptRate < adaptRateMax)
                {
                    adaptRate += adaptRateIncrement;
                    adaptRate = std::min(adaptRate, adaptRateMax);
                }

                if (weightBound < weightBoundMax)
                {
                    weightBound += weightBoundIncrement;
                    for (size_t i = 0; i < dimensions; i++)
                    {
                        auto const normalizationFactor = weightNormalization[i];
                        auto const maxWeight = weightBound * normalizationFactor;
                        auto const minWeight = -maxWeight;

                        aln.SetMaxWeight(i, maxWeight);
                        aln.SetMinWeight(i, minWeight);
                    }
                }

                if (allowedSplits < maxSplits && totalSplits == allowedSplits)
                {
                    // the network has reached max allowable size;
                    // we can try to speed things up by reducing number of epochs per iteration
                    if (epochsPerIteration > minEpochsPerIteration)
                    {
                        epochsPerIteration--;
                    }

                    if (smoothRmseDelta < smoothRmseDeltaSplitThreshold)
                    {
                        // we've stopped decreasing the error effectively; 
                        // try increasing size of network...

                        // allow the number of pieces to increase
                        allowedSplits = std::min(allowedSplits + splitIncrement, maxSplits);
                        splitIncrement *= 2; // next time we can double 
                        
                        // require fewer points to split a piece (allow under determined pieces to split)
                        minPointsPerSplit = std::max<size_t>(2, minPointsPerSplit / 2); 

                        // make it harder to allow future increase in pieces
                        smoothRmseDeltaSplitThreshold *= smoothRmseDeltaSplitThresholdDecay; 
                    }
                }

                // check how we're doing on test set
                if (i % 20 == 0)
                {
                    if (testInput.size() > 0)
                    {
                        rmse = CalculateRMSError(aln, cbegin(testExpected), cbegin(testInput), testInput.size());
                        classErrors = CountClassificationErrors(aln, testExpected, testInput);
                        
                        minTestRmse = std::min(rmse, minTestRmse);
                        minTestClassErrors = std::min(classErrors, minTestClassErrors);

                        std::cout 
                            << "[" << digitToClassify << "] test rmse=" << rmse 
                            << " classification errors=" << classErrors 
                            << " (min = " << minTestRmse << " / " << minTestClassErrors << ")" 
                            << std::endl;
                    }
                }

                // export aln every 100 epochs
                if (i % 100 == 0)
                {
                    std::cout << "[" << digitToClassify << "] exporting aln to " << alnPath << std::endl;
                    aln::JsonExport(alnPath, aln);
                }
            }
        }

        std::cout << "[" << digitToClassify << "] exporting aln to " << alnPath << std::endl;
        aln::JsonExport(alnPath, aln);

        classErrors = CountClassificationErrors(aln, trainExpected, trainInput);
        std::cout << "[" << digitToClassify << "] final train rmse=" << rmse << " classification errors=" << classErrors << " lu count=" << aln.LinearUnits().size() << std::endl;

        if (testInput.size() > 0)
        {
            rmse = CalculateRMSError(aln, cbegin(testExpected), cbegin(testInput), testInput.size());
            classErrors = CountClassificationErrors(aln, testExpected, testInput);
            std::cout << "[" << digitToClassify << "] final test rmse=" << rmse << " classification errors=" << classErrors << std::endl;
        }
    }

    template<typename T>
    void Normalize(aln::DataSet<T>& dataSet, aln::DataSet<T> const& normalization)
    {
        assert(dataSet.size() > 0 && normalization.size() == 2);
        assert(dataSet[0].size() == normalization[0].size());

        auto const& offsets = normalization[0];
        auto const& ranges = normalization[1];

        auto const range = aln::make_range(dataSet.size());
        std::for_each_n(
            aln::ParallelUnsequenced,
            cbegin(range),
            range.size(),
            [&](auto const pointIndex)
            {
                auto& dataPoint = dataSet[pointIndex];
                auto const range2 = aln::make_range(dataPoint.size());
                std::for_each_n(
                    aln::ParallelUnsequenced,
                    begin(range2),
                    range2.size(),
                    [&](auto const elementIndex)
                    {
                        auto const offset = offsets[elementIndex];
                        auto const range = ranges[elementIndex];

                        if (range == 0 || (offset == 0 && range == 1))
                            return;

                        auto value = dataPoint[elementIndex];
                        dataPoint[elementIndex] = (value - offset) / range;
                    });
            });
    }

    template <typename T>
    void RunTrain(
        double const desiredRmse,
        size_t const iterations,
        std::string const& dataFolder, 
        std::vector<int> digits, 
        aln::DataSet<T> const& trainSet, 
        aln::DataSet<T> const& testSet,
        aln::DataSet<T> const& normalization)
    {
        // expect labels to be at column 0 of train and test sets
        // ... slice that column off to form disjoint label and input sets

        aln::DataPoint<T> trainLabels;
        aln::DataSet<T> trainInput;
        std::tie(trainLabels, trainInput) = SliceColumn(trainSet, 0);

        aln::DataPoint<T> testLabels;
        aln::DataSet<T> testInput;
        std::tie(testLabels, testInput) = SliceColumn(testSet, 0);

        aln::DataPoint<T> normLabels;
        aln::DataSet<T> normInput;
        std::tie(normLabels, normInput) = SliceColumn(normalization, 0);

        std::for_each(
            aln::ParallelUnsequenced,
            begin(digits),
            end(digits),
            [&](auto digit)
            {
                // digit aln path
                std::stringstream pathString;
                pathString << dataFolder << "/aln-digit-" << digit << ".json";
                auto const alnPath = pathString.str();

                // save minimal network
                auto aln = BuildBinaryClassificationNetwork<T>(784, desiredRmse);
                aln::JsonExport(alnPath, aln);

                TrainDigit<float>(aln, alnPath, digit, desiredRmse, iterations, trainLabels, trainInput, testLabels, testInput, normInput);
            });
    }

    template <typename T>
    aln::DataSet<T> LoadDataSet(std::string const& dataFolder, std::string const& fileName, size_t rowCountHint)
    {
        return aln::ReadDataSet<float>(dataFolder + "/" + fileName, ',', rowCountHint);
    }

    using ConfusionMatrix = std::array<std::array<size_t, 10>, 10>;

    template <typename T>
    ConfusionMatrix RunTest(
        std::string const& dataFolder, 
        aln::DataSet<T> const& dataSet)
    {
        // expect labels to be at column 0 of data set
        // ... slice that column off to form disjoint label and input sets

        aln::DataPoint<T> dataLabels;
        aln::DataSet<T> dataInput;
        std::tie(dataLabels, dataInput) = SliceColumn(dataSet, 0);

        ConfusionMatrix confusion;
        for (size_t i = 0; i < 10; i++)
        {
            for (size_t j = 0; j < 10; j++)
                confusion[i][j] = 0;
        }

        // load evaluator
        auto evaluator = DigitEvaluator<float>(dataFolder);

        // run on test set
        auto evalStart = std::chrono::high_resolution_clock::now();
        auto results = Evaluate(aln::Sequential, evaluator, dataInput);
        auto evalEnd = std::chrono::high_resolution_clock::now();
        auto evalDuration = std::chrono::duration<double>(evalEnd - evalStart);

        assert(results.size() == dataSet.size());

        std::atomic<size_t> classificationErrors{ 0 };
        auto const indexRange = aln::make_range(dataLabels.size());
        std::for_each_n(
            aln::Sequential,
            begin(indexRange),
            indexRange.size(),
            [&](auto const pointIndex)
            {
                auto const expected = static_cast<int>(dataLabels[pointIndex]);
                auto const actual = results[pointIndex];

                confusion[expected][actual]++;
                    
                if (expected != actual)
                {
                    classificationErrors++;
                }
            });

        std::cout
            << std::endl
            << "final data set classification errors = " << classificationErrors
            << " of " << results.size()
            << " (" << 100.0 * (double)classificationErrors / (double)results.size() << "% error rate)"
            << " time " << evalDuration.count() << " s (" << 1000.0 * evalDuration.count() / (double)results.size() << " ms per point)"
            << std::endl;

        std::cout 
            << std::endl
            << "confusion matrix:"
            << std::endl
            << std::endl;

        std::cout << std::setw(8) << ' ';
        for (size_t i = 0; i < 10; i++)
        {
            std::cout << std::setw(8) << i;
        }
        std::cout << std::endl;

        for (size_t i = 0; i < 10; i++)
        {
            std::cout << std::setw(8) << i;
            for (size_t j = 0; j < 10; j++)
            {
                std::cout << std::setw(8) << confusion[i][j];
            }
            std::cout << std::endl;
        }

        return confusion;
    }
}
