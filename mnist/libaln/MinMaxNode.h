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

#include <array>

#include "aln.h"

namespace aln
{
    enum class MinMaxType
    {
        Min = 1,
        Max = 2
    };

    template<typename T>
    class Network;

    /// <summary>
    /// Represents a minimum/maximum function.
    /// </summary>
    /// <remarks>
    /// Instance members are not safe for use across multiple threads without 
    /// external synchronization.
    /// </remarks>
    template<typename T>
    class MinMaxNode final
    {
    public:

        /// <summary>
        /// Constructs a new instance of <see cref="MinMaxNode"/> suitable for deserialization via move assignment.
        /// </summary>
        MinMaxNode()
            : _type{ }
            , _parentIndex{ }
            , _children{ NodeIndex::Invalid(), NodeIndex::Invalid() }
        {
        }

        /// <summary>
        /// Constructs an instance of a <see cref="MinMaxNode"/>.
        /// </summary>
        /// <param name="type"></param>
        /// <param name="parentIndex"></param>
        /// <param name="leftIndex"></param>
        /// <param name="rightIndex"></param>
        MinMaxNode(MinMaxType type, NodeIndex parentIndex, NodeIndex leftIndex, NodeIndex rightIndex)
            : _type{ type }
            , _parentIndex{ parentIndex }
            , _children{ leftIndex, rightIndex }
        {
        }

        /// <summary>
        /// Get index of parent in the network.
        /// </summary>
        /// <remarks>
        /// If ParentIndex().IsRoot() == true, then this node is the root and refers to itself.
        /// </remarks>
        NodeIndex ParentIndex() const
        {
            return _parentIndex;
        }

        std::array<NodeIndex, 2> const& Children() const { return _children; }

        NodeIndex LeftIndex() const { return _children[0]; }
        NodeIndex RightIndex() const { return _children[1]; }

        MinMaxType Type() const { return _type; }

        bool IsMin() const { return _type == MinMaxType::Min; }
        bool IsMax() const { return _type == MinMaxType::Max; }

    private:

        friend class Network<T>;

        void ReplaceParent(NodeIndex parentIndex)
        {
            assert(!parentIndex.IsLinear() && !parentIndex.IsInvalid());
            _parentIndex = parentIndex;
        }

        void ReplaceChild(NodeIndex oldNodeIndex, NodeIndex newNodeIndex)
        {
            assert(!oldNodeIndex.IsInvalid() && !newNodeIndex.IsInvalid());

            if (_children[0] == oldNodeIndex)
            {
                _children[0] = newNodeIndex;
            }
            else
            {
                assert(_children[1] == oldNodeIndex);
                _children[1] = newNodeIndex;
            }
        }

        MinMaxType _type;
        NodeIndex _parentIndex;
        std::array<NodeIndex, 2> _children;
    };
}