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

#include "mnist.h"

#ifdef _MSC_VER
#define stricmp _stricmp
#else
#define stricmp strcasecmp
#endif

//
// you can build this with g++:
// g++ -O3 -std=c++17 -o Driver main.cpp -I ../packages/nlohmann.json.3.10.4/build/native/include/ -ltbb            
// 
// and then do a training run on all digits
// ./mnist ../data train 0 1 2 3 4 5 6 7 8 9
//

// NOTE: this can run really slowly in DEBUG builds; Release mode with optimizations is recommended

int main(int argc, char* argv[])
{
    std::string dataFolder = ".\\";
    if (argc <= 1)
    {
        std::cout << "usage: mnist.exe dataFolder [train digit0 [digit1] ... [digitN]]" << std::endl;
        return -1;
    }
    dataFolder = argv[1];

    std::vector<int> trainDigits;
    bool const train = argc >=3 && stricmp(argv[2], "train") == 0;
    if (train)
    {
        auto const digitCount = argc - 3;
        for (auto i = 0; i < digitCount; i++)
        {
            // parse digit to train
            char* digitString = argv[3 + i];
            int digit;
            if (strlen(digitString) > 1 || (digit = digitString[0] - '0') < 0 || digit > 9)
            {
                std::cout << "usage: mnist.exe dataFolder [train digit0 [digit1] ... [digitN]]" << std::endl;
                std::cout << "invalid digit '" << digitString << "' ... must be in [0, 9]" << std::endl;
                return -1;
            }

            // ignore repeated digits
            auto exists = 0 != std::count_if(begin(trainDigits), end(trainDigits), [digit](auto const existing) { return existing == digit; });
            if (exists)
            {
                std::cout << "ignoring repeated train digit '" << digitString << "'" << std::endl;
            }
            else
            {
                trainDigits.push_back(digit);
            }
        }
    }

    std::cout << "generating intensity-based normalization constants..." << std::endl;
    aln::DataSet<float> normalization = MakeIntensityNormalization<float>(785, 0);

    std::cout << "reading test set..." << std::endl;
    aln::DataSet<float> testSet = mnist::LoadDataSet<float>(dataFolder, "mnist-test.csv", 10000);

    std::cout << "normalizing test set..." << std::endl;
    mnist::Normalize(testSet, normalization);

    if (trainDigits.size() > 0)
    {
        std::cout << "reading training set..." << std::endl;
        aln::DataSet<float> trainSet = mnist::LoadDataSet<float>(dataFolder, "mnist-train.csv", 60000);

        std::cout << "normalizing training set..." << std::endl;
        mnist::Normalize(trainSet, normalization);

        std::cout << "training on digits..." << std::endl;
        mnist::RunTrain(0.05, 1000, dataFolder, trainDigits, trainSet, testSet, normalization);
        mnist::RunTest(dataFolder, trainSet);
    }

    std::cout << "evaluating test set..." << std::endl;
    mnist::RunTest(dataFolder, testSet);
}

