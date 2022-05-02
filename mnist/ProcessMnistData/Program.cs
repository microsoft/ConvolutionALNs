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
using System.Diagnostics;
using System.IO;

namespace ProcessMnistData
{
    class Program
    {
        static void Main(string[] args)
        {
            // from data at http://yann.lecun.com/exdb/mnist/

            var trainImages = new IdxFile(@"D:\dev\ALNEval\mnist\train-images.idx3-ubyte");
            var trainLabels = new IdxFile(@"D:\dev\ALNEval\mnist\train-labels.idx1-ubyte");
            ExportCsv(@"D:\dev\ALNEval\mnist\mnist-train.csv", trainImages, trainLabels, @"D:\dev\ALNEval\mnist\mnist-train.normalization.csv");


            var testImages = new IdxFile(@"D:\dev\ALNEval\mnist\t10k-images.idx3-ubyte");
            var testLabels = new IdxFile(@"D:\dev\ALNEval\mnist\t10k-labels.idx1-ubyte");
            ExportCsv(@"D:\dev\ALNEval\mnist\mnist-test.csv", testImages, testLabels, null);
        }

        private static void ExportCsv(string path, IdxFile imageFile, IdxFile labelFile, string normalizationPath)
        {
            if (imageFile.Dimensions[0] != labelFile.Dimensions[0])
                throw new ArgumentException();

            var labels = labelFile.Read1dRow();

            var rowCount = imageFile.Dimensions[0];
            var elementsPerRow = 1 + imageFile.Dimensions[1] * imageFile.Dimensions[2];

            double[] mean = null;
            double[] stdev = null;
            if (normalizationPath != null)
            {
                mean = new double[elementsPerRow];
                stdev = new double[elementsPerRow];

                // no normalization for output
                mean[0] = 0;
                stdev[0] = 1;
            }

            using (var writer = File.CreateText(path))
            {
                for (var i = 0; i < rowCount; i++)
                {
                    var label = labels[i];
                    var image = imageFile.Read2dImage(i);
                    Debug.Assert(image.Length == elementsPerRow - 1);

                    var rowValues = new double[elementsPerRow];
                    rowValues[0] = label;

                    for (var j = 0; j < image.Length; j++)
                    {
                        var value = (double)image[j];
                        rowValues[1 + j] = value;
                        
                        if (normalizationPath != null)
                        {
                            mean[1 + j] += value;
                        }
                    }

                    var row = string.Join(',', rowValues);
                    writer.WriteLine(row);
                }
            }

            if (normalizationPath != null)
            {
                for (var j = 1; j < elementsPerRow; j++)
                    mean[j] /= rowCount;

                for (var i = 0; i < rowCount; i++)
                {
                    var image = imageFile.Read2dImage(i);
                    Debug.Assert(image.Length == elementsPerRow - 1);
                    
                    for (var j = 0; j < image.Length; j++)
                    {
                        var value = (double)image[j];

                        var diff = value - mean[1 + j];
                        stdev[1 + j] += diff * diff;
                    }
                }

                for (var j = 1; j < elementsPerRow; j++)
                    stdev[j] = Math.Sqrt(stdev[j] / (rowCount - 1));

                using (var writer = File.CreateText(normalizationPath))
                {
                    var line1 = string.Join(',', mean);
                    var line2 = string.Join(',', stdev);
                    writer.WriteLine(line1);
                    writer.WriteLine(line2);
                }
            }
        }
    }
}
