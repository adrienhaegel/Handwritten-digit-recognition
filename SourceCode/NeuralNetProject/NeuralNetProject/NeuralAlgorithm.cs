
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace OCRMNISTReader
{
    class NeuralAlgorithm
    {
        //This is the main function for the Neural Net

        //static DATA
        public static MNISTReader.DigitImage[] TestImages;
        public static MNISTReader.DigitImage[] TrainingImages;
        public static MNISTReader.DigitImage[] ValidationImages;



        public static void Main(string[] args)
        {
            //The MNIST data have to be in a data folder in the root of the program
            string path = Directory.GetCurrentDirectory().ToString() + "\\data\\";
            //Loading DATA
            TestImages = MNISTReader.ReadTest(path);
            TrainingImages = MNISTReader.ReadTraining(path);
            ValidationImages = MNISTReader.ReadValidation(path);

            Parameters parameters = new Parameters();

            //Defining the default parameters
            parameters.layers_size = new int[] { 784, 100, 10 };
            parameters.nblayers = parameters.layers_size.Length;
            parameters.eta = 1;
            parameters.batchsize = 10;
            parameters.costfunction = Parameters.Costfunction.CROSSENTROPY;
            parameters.regularization = true;
            parameters.lambda = 3;
            parameters.stopafternbsteps = 20;

            //Ask the user to change the parameters
            ConsoleManager(parameters);

            ///////////MAIN FUNCTION //////////////
            Driver(parameters);
            ///////////////////////////////////////

            //pause
            Console.Read();
        }

        // Handles all the calculations
        public static void Driver(Parameters p)
        {
            //for computation time
            DateTime starttime = DateTime.Now;

            //For the logs 
            List<String> log = new List<string>();
            log.Add(p.ToString());

            //Creates a brain on the given parameters
            Brain brain = new Brain(p);


            double bestvalidationresult = 0;
            int step = 1;

            //Computes initial validation error
            Console.Write("Initial validation error :   ");
            double initval = 1 - GetValidationResult(brain);
            Console.WriteLine(initval);
            log.Add((initval.ToString()));

            //For the learning rate update
            int tolerance = 2;
            int stopcounter = 0;

            //Driver
            while (stopcounter < p.stopafternbsteps)
            {
                //LEARNING
                IterateStochasticGradient(1, p.batchsize, brain);

                //TESTING
                double validationresult = GetValidationResult(brain);
                Console.WriteLine("Epoch nb : " + step + "  eta =  " + p.eta.ToString() + "  |    Validation error : " + (1 - validationresult).ToString());
                log.Add(step.ToString() + " " + p.eta.ToString() + " " + (1 - validationresult).ToString());

                //Continues, reduces the learning rate or stops, depending on the result
                if (validationresult > bestvalidationresult)
                {
                    bestvalidationresult = validationresult;
                    tolerance = 2;
                    stopcounter = 0;
                }
                else
                {
                    if (tolerance > 0)
                    {
                        tolerance -= 1;
                    }
                    else
                    {
                        p.eta /= 2;
                        tolerance = 2;
                    }
                    stopcounter += 1;
                }
                step += 1;
            }
            log.Add(" ");
            //END


            //Computing final results
            double finalvalres = (1 - GetValidationResult(brain));
            double finaltestres = (1 - GetTestResultandPrint(brain));

            //Writing logs
            Console.WriteLine("|||||||   Final Validation Error : " + finalvalres + "     ||||||| ");
            Console.WriteLine("|||||||   Final Test Error : " + finaltestres + "     ||||||| ");
            log.Add(finalvalres.ToString());
            log.Add(finaltestres.ToString());
            TimeSpan duration = DateTime.Now - starttime;
            log.Add(duration.TotalSeconds.ToString());
            WriteLog(log);

        }

        //This is only to write the logs collected during the algorithm
        private static void WriteLog(List<string> log)
        {
            string date = DateTime.Now.Day + "_" + DateTime.Now.Hour + "_" + DateTime.Now.Minute;

            string namefile = Directory.GetCurrentDirectory() + "\\" + date + "\\log.txt";
            Directory.CreateDirectory(Directory.GetCurrentDirectory() + "\\" + date);

            using (System.IO.StreamWriter file =
            new System.IO.StreamWriter(@namefile))
            {
                foreach (string line in log)
                {
                    file.WriteLine(line);
                }
            }
        }


        //USER INTERFACE
        public static void ConsoleManager(Parameters p)
        {
            Console.WriteLine(" ||||||||||  Welcome in MNIST Neural Networks ||||||||||");
            Console.WriteLine(" This code has been written by Haegel Adrien, KAIST, 20154913 ");
            Console.WriteLine(" ");
            Console.WriteLine("To launch the default neural network, press ENTER");
            Console.WriteLine("To modify the neural net parameters, press m");

            ConsoleKeyInfo key = Console.ReadKey();
            while ((key.Key != ConsoleKey.Enter) && (key.KeyChar != 'm'))
            {
                key = Console.ReadKey();

            }

            if (key.Key == ConsoleKey.Enter)
            {
                Console.WriteLine("The default values has been set : Layers : 784-100-10, batch size =10, lambda = 2.5");
                Console.WriteLine("");
                Console.WriteLine("Thank you, the network is now going to start learning!");
                Console.WriteLine("");
            }
            else if (key.KeyChar == 'm')
            {
                Console.WriteLine("");
                Console.WriteLine("Enter the number of layers");
                int nblayers = (int.Parse(Console.ReadLine()));
                Console.WriteLine("First layer is size 784");
                int[] layers = new int[nblayers];
                p.nblayers = nblayers;

                layers[0] = 784;
                layers[nblayers - 1] = 10;

                for (int i = 1; i < nblayers - 1; i++)
                {
                    Console.WriteLine("Please enter size of next layer : Layer " + (i + 1));
                    layers[i] = (int.Parse(Console.ReadLine()));
                }

                Console.WriteLine("Last layer is size 10.");
                Console.WriteLine("");
                p.layers_size = layers;
                Console.WriteLine("Please enter batch size (recommended = 10) ");
                p.batchsize = (int.Parse(Console.ReadLine()));

                Console.WriteLine("Please enter regularization value (recommended = 2.5) ");
                string lambd = Console.ReadLine();
                p.lambda = double.Parse(lambd.Replace('.', ','));
                Console.WriteLine("");
                Console.WriteLine("Thank you, the network is now going to start learning!");
                Console.WriteLine("");
            }
        }


        //Iterate the stochastic gradient algorithm a number of times
        public static void IterateStochasticGradient(int nbiter, int batchsize, Brain brain)
        {
            for (int i = 0; i < nbiter; i++)
            {
                StochasticGradient(batchsize, brain);
            }
        }

        //Stochastic Gradient Descent algortithm
        public static void StochasticGradient(int batchsize, Brain brain)
        {
            //Take data
            MNISTReader.DigitImage image;
            List<MNISTReader.DigitImage> samples = TrainingImages.ToList();
            //Shuffle it
            Shuffle(samples);
            //For each epoch:
            while (samples.Count != 0)
            {
                //Treat "batchsize" number of samples
                for (int i = 0; i < batchsize; i++)
                {
                    //For each sample
                    if (samples.Count > 0)
                    {

                        image = samples[0];
                        samples.RemoveAt(0);
                        Vector arrayvector = ConvertImageToVector(image);
                        //Input to the network
                        brain.SetInitialActivation(arrayvector);
                        //Forward propagation
                        brain.FeedForward();
                        //Backpropagation
                        Vector result = new Vector(10);
                        result.data[image.label] = 1;
                        brain.BackPropagation(result);
                    }
                }
                //Update the weights
                brain.Update(batchsize);
            }
        }

        //Shuffles the Dataset (used to have random data samples)
        public static void Shuffle(List<MNISTReader.DigitImage> list)
        {
            Random rng = new Random();
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                MNISTReader.DigitImage value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }


        //Compute the result on the test data
        public static double GetTestResult(Brain brain)
        {
            int success = 0;
            int total = 0;
            foreach (var image in TestImages)
            {
                Vector doublevector = ConvertImageToVector(image);
                brain.SetInitialActivation(doublevector);
                brain.FeedForward();
                int result = brain.GetResult();

                if (result == image.label)
                {
                    success += 1;
                }
                total += 1;

                if (total == 10000)
                {
                    break;
                }

            }
            return (double)success / (double)total;
        }

       // Compute the result on the test data and print the final images
        public static double GetTestResultandPrint(Brain brain)
        {
            string correctpath = Directory.GetCurrentDirectory() + "\\Resultimages\\Correct\\";
            string errorpath = Directory.GetCurrentDirectory() + "\\Resultimages\\Error\\";

            Directory.CreateDirectory(correctpath);
            Directory.CreateDirectory(errorpath);

            int success = 0;
            int total = 0;
            int index = 0;
            foreach (var image in TestImages)
            {
                Vector doublevector = ConvertImageToVector(image);
                brain.SetInitialActivation(doublevector);
                brain.FeedForward();
                int result = brain.GetResult();

                if (result == image.label)
                {
                    success += 1;
                    ByteArrayToImageFilebyMemoryStream(image.pixels, correctpath, result, index);
                }
                else
                {
                    ByteArrayToImageFilebyMemoryStream(image.pixels, errorpath, result, index);
                }
                total += 1;

                if (total == 10000)
                {
                    break;
                }
                index += 1;
            }
            return (double)success / (double)total;
        }

        //Compute the result on the validation data
        public static double GetValidationResult(Brain brain)
        {
            int success = 0;
            int total = 0;
            foreach (var image in ValidationImages)
            {
                Vector doublevector = ConvertImageToVector(image);
                brain.SetInitialActivation(doublevector);
                brain.FeedForward();
                int result = brain.GetResult();

                if (result == image.label)
                {
                    success += 1;
                }
                total += 1;

                if (total == 10000)
                {
                    break;
                }

            }
            return (double)success / (double)total;
        }


        //MNIST to vector converter
        public static Vector ConvertImageToVector(MNISTReader.DigitImage image)
        {
            Vector doublevector = new Vector(28 * 28);
            int count = 0;
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    if (image.pixels[i][j] != 0)
                    {
                        doublevector.data[count] = image.pixels[i][j] / 256.0;
                    }

                    count++;
                }
            }
            return doublevector;
        }

        //Printing an image to bitmap
        public static void ByteArrayToImageFilebyMemoryStream(byte[][] imageByte, string path, int recognition, int index)
        {
            string filepath = path + recognition.ToString() + "_" + index + ".jpg";

            int width = 28;
            int height = 28;

            Bitmap Image = new Bitmap(width, height);

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    int pix = 255 - imageByte[j][i];

                    Color myColor = Color.FromArgb(pix, pix, pix);
                    Image.SetPixel(i, j, myColor);
                }

            Image.Save(filepath);
        }

    }
}
