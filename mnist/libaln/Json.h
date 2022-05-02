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

#include <iomanip>
#include <vector>

#include "aln.h"
#include "nlohmann/json.hpp"

namespace aln
{
    namespace details
    {
        template <typename RandomAccessIterator>
        struct VectorSegment
        {
            VectorSegment(RandomAccessIterator first, size_t count)
                : First{ first }
                , Count{ count }
            {
            }

            RandomAccessIterator First;
            size_t Count;
        };

        template<typename RandomAccessIterator>
        VectorSegment<RandomAccessIterator> make_segment(RandomAccessIterator first, size_t count)
        {
            return VectorSegment<RandomAccessIterator>(first, count);
        }

        template<typename RandomAccessIterator>
        inline void to_json(nlohmann::json& j, VectorSegment<RandomAccessIterator> const& segment)
        {
            j = nlohmann::json::array();
            std::copy_n(
                segment.First, 
                segment.Count,
                std::back_inserter(j));
        }
    }

    inline void to_json(nlohmann::json& j, NodeIndex const& value)
    {
        if (value.IsInvalid())
        {
            j = nullptr;
        }
        else
        {
            j = nlohmann::json
            {
                { "type", (value.IsLinear() ? "Linear" : "MinMax") },
                { "index", value.Ordinal() }
            };
        }
    }

    inline void from_json(nlohmann::json const& j, NodeIndex & value)
    {
        if (j.is_null())
        {
            value = NodeIndex::Invalid();
        }
        else
        {
            auto const type = j.at("type").get<std::string>();
            auto const ordinal = j.at("index").get<size_t>();
            if (type == "Linear")
                value = NodeIndex::Linear(ordinal);
            else if (type == "MinMax")
                value = NodeIndex::MinMax(ordinal);
            else
                throw std::invalid_argument("invalid NodeIndex json value");
        }
    }

    inline void to_json(nlohmann::json& j, MinMaxType const& value)
    {
        j = value == MinMaxType::Min 
            ? "Min" 
            : "Max";
    }

    inline void from_json(nlohmann::json const& j, MinMaxType& value)
    {
        auto string = j.get<std::string>();
        if (string == "Max")
            value = MinMaxType::Max;
        else if (string == "Min")
            value = MinMaxType::Min;
        else
            throw std::invalid_argument("invalid MinmaxType json value");
    }

    template<typename T>
    inline void to_json(nlohmann::json& j, FixedVector<T> const& value)
    {
        j = nlohmann::json::array();
        std::copy(
            cbegin(value),
            cend(value),
            std::back_inserter(j));
    }

    template<typename T>
    inline void from_json(nlohmann::json const& j, FixedVector<T>& value)
    {
        auto array = j.get<std::vector<T>>();
        if (array.size() != value.size())
            throw std::invalid_argument("invalid FixedVector json value: wrong size");

        std::copy(begin(array), end(array), begin(value));
    }

    template<typename T>
    inline void to_json(nlohmann::json& j, LinearUnit<T> const& value)
    {
        auto const& w = value.W();

        j = nlohmann::json
        {
            { "parent",      value.ParentIndex() },
            { "bias",        value.Bias() },
            { "weights",     w },
        };
    }

    template<typename T>
    inline void from_json(nlohmann::json const& j, LinearUnit<T>& value)
    {
        auto parent = j.at("parent").get<NodeIndex>();
        auto bias = j.at("bias").get<T>();
        auto weights = j.at("weights").get<std::vector<T>>();

        auto dimensions = weights.size();
        value = std::move(LinearUnit<T>(dimensions, parent));
        
        // we didn't save centroid or distance vectors, 
        // so this node is constant and can't be adapted
        value.SetIsConstant(true);
        value.SetBias(bias);

        auto& w = value.W();
        std::copy(begin(weights), end(weights), begin(w));
    }

    template<typename T>
    inline void to_json(nlohmann::json& j, MinMaxNode<T> const& value)
    {
        j = nlohmann::json
        {
            { "parent", value.ParentIndex() },
            { "type",   value.Type() },
            { "left",   value.LeftIndex() },
            { "right",  value.RightIndex() }
        };
    }

    template<typename T>
    inline void from_json(nlohmann::json const& j, MinMaxNode<T>& value)
    {
        auto parent = j.at("parent").get<NodeIndex>();
        auto type = j.at("type").get<MinMaxType>();
        auto left = j.at("left").get<NodeIndex>();
        auto right = j.at("right").get<NodeIndex>();

        value = std::move(MinMaxNode<T>(type, parent, left, right));
    }

    template<typename T>
    inline void to_json(nlohmann::json& j, Network<T> const& value)
    {
        j = nlohmann::json
        {
            { "dimensions", value.Dimensions() },
            { "outputNode",   value.RootIndex() },
            { "minMaxNodes",  value.MinMaxNodes() },
            { "linearUnits",  value.LinearUnits() }
        };
    }

    template<typename T>
    inline void from_json(nlohmann::json const& j, Network<T>& value)
    {
        auto dimensions = j.at("dimensions").get<size_t>();
        auto rootIndex = j.at("outputNode").get<NodeIndex>();
        auto minMaxNodes = j.at("minMaxNodes").get<std::vector<MinMaxNode<T>>>();
        auto linearUnits = j.at("linearUnits").get<std::vector<LinearUnit<T>>>();

        
        // todo: validate?

        value = std::move(Network<T>(dimensions, rootIndex, minMaxNodes, linearUnits));
    }

    template<typename T>
    nlohmann::json ToJson(Network<T> const& aln)
    {
        static_assert(std::is_floating_point_v<T>);

        nlohmann::json j = aln;

        return j;
    }

    template<typename T>
    void JsonExport(std::ostream& stream, Network<T> const& aln)
    {
        static_assert(std::is_floating_point_v<T>);

        nlohmann::json j = aln;

        stream << std::setw(4) << j;
        stream.flush();
    }

    template<typename T>
    void JsonExport(std::string fileName, Network<T> const& aln)
    {
        std::ofstream stream(fileName, std::ofstream::trunc);
        JsonExport(stream, aln);
    }

    template<typename T>
    Network<T> JsonImport(std::istream& stream)
    {
        static_assert(std::is_floating_point_v<T>);

        nlohmann::json j;
        stream >> j;

        return j.get<Network<T>>();
    }

    template<typename T>
    Network<T> JsonImport(std::string fileName)
    {
        std::ifstream stream(fileName);
        return JsonImport<T>(stream);
    }

}

