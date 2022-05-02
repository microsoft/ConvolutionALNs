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

#include <cstdlib>
#include <fstream>
#include <functional>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "aln.h"

namespace aln
{
    namespace details
    {
        inline std::string& ltrim(std::string& str)
        {
            auto it2 = std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace<char>(ch, std::locale::classic()); });
            str.erase(str.begin(), it2);
            return str;
        }

        inline std::string& rtrim(std::string& str)
        {
            auto it1 = std::find_if(str.rbegin(), str.rend(), [](char ch) { return !std::isspace<char>(ch, std::locale::classic()); });
            str.erase(it1.base(), str.end());
            return str;
        }

        inline std::string& trim(std::string& str)
        {
            return ltrim(rtrim(str));
        }
    }

    template <typename T>
    using DataPoint = std::vector<T>;

    template <typename T>
    using DataSet = std::vector<DataPoint<T>>;

    enum class PointParseResult
    {
        Success,
        Ignore,
        Fail
    };

    template<typename T>
    PointParseResult ParseDataPoint(size_t ordinal, std::string const& toParse, T& value)
    {
        try
        {
            value = static_cast<T>(std::stod(toParse.c_str()));
        }
        catch (std::invalid_argument const&)
        {
            return PointParseResult::Fail;
        }
        
        return PointParseResult::Success;
    }

    template<typename T, typename PointParseFn>
    DataSet<T> ParseDataSet(std::string fileName, char const delimiter, size_t const rowCountHint, size_t maxRows, PointParseFn elementParseFn)
    {
        DataSet<T> dataSet;
        if (rowCountHint > 0)
        {
            dataSet.reserve(rowCountHint);
        }
        size_t elementCountHint = 0;

        auto stream = std::ifstream(fileName);
        while (stream.good() && dataSet.size() < maxRows)
        {
            std::string line;
            std::getline(stream, line);
            line = details::trim(line);
            if (line.empty())
                continue;

            std::istringstream lineStream(line);
            DataPoint<T> point;
            if (elementCountHint > 0)
                point.reserve(elementCountHint);
            
            size_t elementCount = 0;
            while (lineStream.good())
            {
                std::string element;
                std::getline(lineStream, element, delimiter);
                element = details::trim(element);
                if (element.empty())
                {
                    elementCount++;
                    continue;
                }

                T value;
                auto result = elementParseFn(elementCount++, element, value);
                if (result == PointParseResult::Fail)
                {
                    throw std::logic_error("error in data file");
                }
                else if (result == PointParseResult::Success)
                {
                    point.push_back(value);
                }

                if (lineStream.eof() && point.size() > 0)
                {
                    elementCountHint = (std::max)(elementCountHint, point.size());

                    dataSet.push_back(std::move(point));
                    break;
                }
                else if (lineStream.fail())
                {
                    throw std::logic_error("error in data file");
                }
            }
        }

        if (dataSet.size() > 0)
        {
            auto const dimensions = dataSet[0].size();
            for (size_t i = 1; i < dataSet.size(); i++)
            {
                if (dataSet[i].size() != dimensions)
                    throw std::logic_error("mismatched dimensions in data file");
            }
        }

        return dataSet;
    }

    template<typename T>
    DataSet<T> ReadDataSet(std::string const fileName, char const delimiter, size_t const rowCountHint, size_t maxRows = std::numeric_limits<size_t>::max())
    {
        return ParseDataSet<T>(std::move(fileName), delimiter, rowCountHint, maxRows, ParseDataPoint<T>);
    }
}
