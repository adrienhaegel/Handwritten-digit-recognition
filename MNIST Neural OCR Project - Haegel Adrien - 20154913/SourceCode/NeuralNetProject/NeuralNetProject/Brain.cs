
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;




using System.IO;



namespace OCRMNISTReader
{
//Holds the structure of the neural network
    class Brain
    {
        public NeuronLayer[] layers;
        public int nblayers;

        //Construct the network
        public Brain(Parameters parameters)
        {
            NeuronLayer.parameters = parameters;
            AxonLayer.parameters = parameters;

            int[] layerssize = parameters.layers_size;
            nblayers = layerssize.Length;
            layers = new NeuronLayer[nblayers];

            layers[0] = new NeuronLayer(layerssize[0]);
            

            for (int i = 1; i < layerssize.Length; i++)
            {
                layers[i] = new NeuronLayer(layerssize[i],layers[i-1]);
            }
        }

        //Set the INPUT (it will be an image)
        public void SetInitialActivation(Vector values)
        {
            layers[0].SetActivation(values);
        }

        //feedforward from the first layers to the last one
        public void FeedForward()
        {
            for (int i = 1; i < nblayers; i++)
            {
                layers[i].FeedForward();
            }

        }

        //Backpropagate from the last layer to the first one
        public void BackPropagation(Vector result)
        {
            layers[nblayers - 1].Compute_Output_Error(result);
            for (int i = nblayers-2; i > 0; i--)
            {
                layers[i].Backpropagate();
            }
        }

        //After Forward and backward on the entire batch, update
        public void Update(int batchsize)
        {
            for (int i = nblayers - 1; i > 0; i--)
            {
                layers[i].Update_Bias_and_Weights(batchsize);
            }
        }

        //Get the result of the propagatioin (result = guess for mnist)
        public int GetResult()
        {
            return this.layers[nblayers - 1].GetResult();
        }
    }
}
