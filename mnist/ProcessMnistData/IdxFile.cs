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

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;

namespace ProcessMnistData
{

    // based on http://yann.lecun.com/exdb/mnist/

    class IdxFile : IDisposable
    {
        readonly FileStream _file;
        readonly int[] _dimensions;

        public IdxFile(string path)
        {
            _file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            _dimensions = ReadHeader();
        }

        public void Dispose()
        {
            _file.Dispose();
        }

        public void Close()
        {
            _file.Close();
        }

        public IReadOnlyList<int> Dimensions => _dimensions;

        // reads the entire row stored in a one dimensional index
        public byte[] Read1dRow()
        {
            if (_dimensions.Length != 1)
                throw new InvalidOperationException("not a 1 dimensional index");

            var offset = 4L + _dimensions.Length * 4L;
            _file.Seek(offset, SeekOrigin.Begin);

            var byteCount = _dimensions[0];
            var buffer = new byte[byteCount];
            _file.ReadComplete(buffer, 0, buffer.Length);

            return buffer;
        }

        // reads image in row major order
        public byte[] Read2dImage(int imageIndex)
        {
            if (_dimensions.Length != 3)
                throw new InvalidOperationException("not a 3 dimensional index");

            var imageCount = _dimensions[0];
            if (imageIndex < 0 || imageIndex >= imageCount)
                throw new ArgumentOutOfRangeException();

            var imageSize = _dimensions[1] * _dimensions[2]; // rows * columns

            var offset = (4L + _dimensions.Length * 4L) + (imageIndex * imageSize);
            _file.Seek(offset, SeekOrigin.Begin);

            var buffer = new byte[imageSize];
            _file.ReadComplete(buffer, 0, buffer.Length);

            return buffer;
        }

        private int[] ReadHeader()
        {
            _file.Seek(0, SeekOrigin.Begin);

            var buffer = new byte[4];
            _file.ReadComplete(buffer, 0, 4);

            if (buffer[0] != 0 || buffer[1] != 0 || buffer[2] != 0x08 || buffer[3] == 0)
                throw new InvalidOperationException("Unexpected idx file header");

            var count = buffer[3];
            var dimensions = new int[count];
            for (var i = 0; i < count; i++)
            {
                _file.ReadComplete(buffer, 0, 4);
                
                // element count stored big-endian
                var elements = buffer[0] << 24
                    | buffer[1] << 16
                    | buffer[2] << 8
                    | buffer[3];

                dimensions[i] = elements;
            }

            return dimensions;
        }
    }
}
