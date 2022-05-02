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

#include <unordered_map>
#include "aln.h"

namespace aln
{
    /// <summary>
    /// EvaluationRoute is useful in the context of an evaluation during training to test which 
    /// inputs to evaluate first, by tracking which child, if any, of a node was responsible for 
    /// the network's value during a previous evaluation on the same input.  
    /// </summary>
    class EvaluationRoute final
    {
    public:

        bool IsEmpty() const
        {
            return _routeMap.empty();
        }

        void Clear()
        {
            _routeMap.clear();
        }

        /// <summary>
        /// Updates an evaluation route from the specified root index and evaluation result.
        /// </summary>
        template<typename T>
        void Update(NodeIndex rootIndex, EvaluationResult<T> const& result)
        {
            // starting from root index, create a mapping from responsible parent to responsible child
            auto const& responsibilities = result.ResponsibleChildren();
            auto parentIndex = rootIndex;
            while (!parentIndex.IsLinear())
            {
                auto const childIndex = responsibilities[parentIndex.Ordinal()];
                _routeMap.insert_or_assign(parentIndex, childIndex);
                parentIndex = childIndex;
            }
        }

        /// <summary>
        /// Gets the responsible child index of the specified parent index.
        /// </summary>
        /// <param name="index"></param>
        /// <returns>The responsible child index if <paramref name="parentIndex"/> is in the evaluation route.</returns>
        std::optional<NodeIndex> GetResponsibleChildOf(NodeIndex parentIndex) const
        {
            auto it = _routeMap.find(parentIndex);
            if (it == end(_routeMap))
                return {};

            return it->second;
        }


    private:
        std::unordered_map<NodeIndex, NodeIndex> _routeMap;

    };


}
