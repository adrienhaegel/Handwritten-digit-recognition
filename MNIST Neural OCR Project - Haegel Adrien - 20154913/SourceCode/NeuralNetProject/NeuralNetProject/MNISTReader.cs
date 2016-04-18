using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OCRMNISTReader
{
    ////  This class treats the MNIST data /////
    public static class MNISTReader
    {
        public static DigitImage[] ReadTest(string path)
        {
            try
            {
                FileStream ifsLabels =
                 new FileStream(@path+"t10k-labels.idx1-ubyte",
                 FileMode.Open); // test labels
                FileStream ifsImages =
                 new FileStream(@path + "t10k-images.idx3-ubyte",
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                DigitImage[] TestImages = new DigitImage[10000];

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                // each test image
                for (int di = 0; di < 10000; ++di)
                {
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();

                    DigitImage dImage =
                      new DigitImage(pixels, lbl);



                    TestImages[di] = dImage;
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                return TestImages;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
                return null;
            }

        }


        public static DigitImage[] ReadTraining(string path)
        {
            try
            {

                FileStream ifsLabels =
                 new FileStream(@path + "train-labels.idx1-ubyte",
                 FileMode.Open); // test labels
                FileStream ifsImages =
                 new FileStream(@path + "train-images.idx3-ubyte",
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                DigitImage[] TestImages = new DigitImage[60000];

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                // each test image
                for (int di = 0; di < 60000; ++di)
                {
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();

                    DigitImage dImage =
                      new DigitImage(pixels, lbl);



                    TestImages[di] = dImage;
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                TestImages = TestImages.Take(50000).ToArray();

                return TestImages;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
                return null;
            }

        }



        public static DigitImage[] ReadValidation(string path)
        {
            try
            {

                FileStream ifsLabels =
                 new FileStream(@path + "train-labels.idx1-ubyte",
                 FileMode.Open); // test labels
                FileStream ifsImages =
                 new FileStream(@path + "train-images.idx3-ubyte",
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                DigitImage[] ValidationImages = new DigitImage[60000];

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                // each test image
                for (int di = 0; di < 60000; ++di)
                {
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();

                    DigitImage dImage =
                      new DigitImage(pixels, lbl);



                    ValidationImages[di] = dImage;
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();
                ValidationImages = ValidationImages.Reverse().Take(10000).Reverse().ToArray();
                return ValidationImages;
            }

            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
                return null;
            }

        }

        public class DigitImage
        {
            public byte[][] pixels;
            public byte label;

            public DigitImage(byte[][] pixels,
              byte label)
            {
                this.pixels = new byte[28][];
                for (int i = 0; i < this.pixels.Length; ++i)
                    this.pixels[i] = new byte[28];

                for (int i = 0; i < 28; ++i)
                    for (int j = 0; j < 28; ++j)
                        this.pixels[i][j] = pixels[i][j];

                this.label = label;
            }

            public override string ToString()
            {
                string s = "";
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        if (this.pixels[i][j] == 0)
                            s += " "; // white
                        else if (this.pixels[i][j] == 255)
                            s += "O"; // black
                        else
                            s += "."; // gray
                    }
                    s += "\n";
                }
                s += this.label.ToString();
                return s;
            } // ToString

        }
    }
}
