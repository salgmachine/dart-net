package com.example.demo;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This code example is featured in this youtube video
 * https://www.youtube.com/watch?v=ECA6y6ahH5E
 *
 * This differs slightly from the Video Example, The Video example had the data
 * already downloaded This example includes code that downloads the data
 *
 * Data is downloaded from wget
 * http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz followed by
 * tar xzvf mnist_png.tar.gz
 *
 * This examples builds on the MnistImagePipelineExample by adding a Neural Net
 */
public class MnistImagePipelineExampleAddNeuralNet {
	private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleAddNeuralNet.class);

	/** Data URL for downloading */
	public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

	/** Location to save and extract the training/testing data */
	public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

	public static void main(String[] args) throws Exception {
		// image information
		// 28 * 28 grayscale
		// grayscale implies single channel
		int height = 59;
		int width = 50;
		int channels = 1;
		int rngseed = 123;
		Random randNumGen = new Random(rngseed);
		int batchSize = 2;
		int outputNum = 3;
		int numEpochs = 3;
		double rate = 0.0015;

		/*
		 * This class downloadData() downloads the data stores the data in java's tmpdir
		 * 15MB download compressed It will take 158MB of space when uncompressed The
		 * data can be downloaded manually here
		 * http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
		 */
//    MnistImagePipelineExample.downloadData();

		// Define the File Paths
		File trainData = new File("src/main/resources/data/training");
		File testData = new File("src/main/resources/data/testing");
		File predictData = new File("src/main/resources/data/predict");
		

		// Define the FileSplit(PATH, ALLOWED FORMATS,random)
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, null);
		FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, null);
		FileSplit predict = new FileSplit(predictData, NativeImageLoader.ALLOWED_FORMATS, null);

		// Extract the parent path as the image label

		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());

		// Initialize the record reader
		// add a listener, to extract the name
		recordReader.initialize(train);
		recordReader.setListeners(new LogRecordListener());

		// DataSet Iterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

		// Scale pixel values to 0-1
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);

		log.info("record reader -labels {}", recordReader.getLabels());
		
		
		dataIter.forEachRemaining(ds -> {
			log.info("dataset labels : {}", ds.getLabels().toString());
		});
		dataIter.reset();

		// Build Our Neural Network
		log.info("BUILD MODEL");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngseed) // include a random seed for
																							// reproducibility
				.activation(Activation.RELU).weightInit(WeightInit.XAVIER).updater(new Nesterovs(rate, 0.98))
				.l2(rate * 0.005) // regularize learning model.list()
				.regularization(true).list()
//				.layer(0,
//						new DenseLayer.Builder().nIn(height * width).nOut(180).activation(Activation.RELU)
//								.weightInit(WeightInit.XAVIER).build())
//				.layer(1,
//						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(180)
//								.nOut(outputNum).activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build())
				.layer(0, new DenseLayer.Builder() // create the first input layer.
						.nIn(width * height).nOut(500).build())
				.layer(1, new DenseLayer.Builder() // create the second input layer
						.nIn(500).nOut(100).build())
				.layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create hidden layer
						.activation(Activation.SOFTMAX).nIn(100).nOut(outputNum).build())
				.setInputType(InputType.convolutional(height, width, channels))
				.inferenceWorkspaceMode(WorkspaceMode.SINGLE).trainingWorkspaceMode(WorkspaceMode.SINGLE).backprop(true)
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		// The Score iteration Listener will log
		// output to show how well the network is training
		model.setListeners(new ScoreIterationListener(10));

		log.info("TRAIN MODEL");
		dataIter.forEachRemaining(ds -> {
			model.fit(ds);
		});
		log.info("summary {}", model.summary());

		for (int i = 0; i < 15; i++) {

			model.fit(dataIter);
			System.out.println("score " + model.score());
		}

		log.info("EVALUATE MODEL");

		// The model trained on the training dataset split
		// now that it has trained we evaluate against the
		// test data of images the network has not seen

		recordReader.initialize(test);
		recordReader.setListeners(new LogRecordListener());
		DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		scaler.fit(testIter);
		testIter.setPreProcessor(scaler);

		/*
		 * log the order of the labels for later use In previous versions the label
		 * order was consistent, but random In current verions label order is
		 * lexicographic preserving the RecordReader Labels order is no longer needed
		 * left in for demonstration purposes
		 */
		List<String> labels = recordReader.getLabels();
		log.info("test labels {}", labels);

		// Create Eval object with 10 possible classes
		Evaluation eval = new Evaluation(outputNum);

		System.out.println("Evaluate model....");
		while (testIter.hasNext()) {
			DataSet t = testIter.next();
			INDArray features = t.getFeatures();
			INDArray lables = t.getLabels();
			INDArray predicted = model.output(features, false);

			eval.eval(lables, predicted);

		}

		// Print the evaluation statistics
		System.out.println(eval.stats());

		log.info("stats {}", eval.stats());
		log.info("confusion matrix {}", eval.getConfusion().getMatrix());
		log.info("label list {}", eval.getLabelsList());
		
		
		
		ImageRecordReader predictRecordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
		predictRecordReader.initialize(predict);
		predictRecordReader.setListeners(new LogRecordListener());
		DataSetIterator predictIter = new RecordReaderDataSetIterator(predictRecordReader, batchSize, 1, 1);
		scaler.fit(predictIter);
		predictIter.setPreProcessor(scaler);
		
		log.info("predict labels : {}", predictRecordReader.getLabels());
		
		
//		recordReader.initialize(predict);
//		recordReader.setListeners(new LogRecordListener());
//		
//		testIter = new RecordReaderDataSetIterator(predictRecordReader, batchSize, 1, outputNum);
//		scaler.fit(testIter);
//		testIter.setPreProcessor(scaler);
//		
//		testIter.forEachRemaining(ds -> {
//			
//			List<String> predict2 = model.predict(ds);
//			log.info("Prediction : {}", predict2);
//		});

	}

}