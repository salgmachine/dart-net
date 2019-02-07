package com.example.demo.nets;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.EqualizeHistTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.example.demo.RunTracker;

/**
 * Created by tom hanlon on 11/7/16. This code example is featured in this
 * youtube video https://www.youtube.com/watch?v=GLC8CIoHDnI
 *
 * This differs slightly from the Video Example The Video example had the data
 * already downloaded This example includes code that downloads the data
 *
 * The Data Directory mnist_png will have two child directories training and
 * testing The training and testing directories will have directories 0-9 with
 * 28 * 28 PNG images of handwritten images
 *
 * The code here shows how to use a ParentPathLabelGenerator to label the images
 * as they are read into the RecordReader
 *
 * The pixel values are scaled to values between 0 and 1 using
 * ImagePreProcessingScaler
 *
 * In this example a loop steps through 3 images and prints the DataSet to the
 * terminal. The expected output is the 28* 28 matrix of scaled pixel values the
 * list with the label for that image and a list of the label values
 *
 * This example also applies a Listener to the RecordReader that logs the path
 * of each image read You would not want to do this in production. The reason it
 * is done here is to show that a handwritten image 3 (for example) was read
 * from directory 3, has a matrix with the shown values, has a label value
 * corresponding to 3
 */
public class MnistNet {
	private static Logger log = LoggerFactory.getLogger(MnistNet.class);

	public void run(String basepath) throws Exception {

		if (RunTracker.getRuns().containsKey("mnist") && RunTracker.getRuns().get("mnist") == true) {
			return;
		}

		RunTracker.getRuns().put("mnist", true);
		
		// Initialize the user interface backend
		UIServer uiServer = UIServer.getInstance();

		// Configure where the network information (gradients, score vs. time etc) is to
		// be stored. Here: store in memory.
		StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File), for saving
																// and loading later

		// Attach the StatsStorage instance to the UI: this allows the contents of the
		// StatsStorage to be visualized
		uiServer.attach(statsStorage);

		/*
		 * image information 28 * 28 grayscale grayscale implies single channel
		 */
		int height = 96;
		int width = 96;
		int channels = 1;
		int rngseed = 123;
		Random randNumGen = new Random(rngseed);
		int batchSize = 1;
		int outputNum = 20;

		/*
		 * This class downloadData() downloads the data stores the data in java's tmpdir
		 * 15MB download compressed It will take 158MB of space when uncompressed The
		 * data can be downloaded manually here
		 * http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
		 */
//    downloadData();

		// Define the File Paths
		File trainData = new File(basepath + "/train");
		File testData = new File(basepath + "/test");

		// Define the FileSplit(PATH, ALLOWED FORMATS,random)
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

		// Extract the parent path as the image label
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

		// Initialize the record reader
		// add a listener, to extract the name
		recordReader.initialize(train, new EqualizeHistTransform());

		// The LogRecordListener will log the path of each image read
		// used here for information purposes,
		// If the whole dataset was ingested this would place 60,000
		// lines in our logs
		// It will show up in the output with this format
		// o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png
		recordReader.setListeners(new LogRecordListener());

		// DataSet Iterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
				recordReader.getLabels().size());

		// Scale pixel values to 0-1
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);

		String trainingLabels = dataIter.getLabels().toString();

		// In production you would loop through all the data
		// in this example the loop is just through 3
		// images for demonstration purposes
//    for (int i = 1; i < 3; i++) {
//      DataSet ds = dataIter.next();
//      log.info(ds.toString());
//      log.info(dataIter.getLabels().toString());
//    }

		ConvolutionLayer layer0 = new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(16).stride(1, 1).padding(2, 2)
				.weightInit(WeightInit.XAVIER).name("First convolution layer").activation(Activation.RELU).build();

		SubsamplingLayer layer1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
				.stride(2, 2).name("First subsampling layer").build();

		ConvolutionLayer layer2 = new ConvolutionLayer.Builder(5, 5).nOut(20).stride(1, 1).padding(2, 2)
				.weightInit(WeightInit.XAVIER).name("Second convolution layer").activation(Activation.RELU).build();

		SubsamplingLayer layer3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
				.stride(2, 2).name("Second subsampling layer").build();

		ConvolutionLayer layer4 = new ConvolutionLayer.Builder(5, 5).nOut(20).stride(1, 1).padding(2, 2)
				.weightInit(WeightInit.XAVIER).name("Third convolution layer").activation(Activation.RELU).build();

		SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
				.stride(2, 2).name("Third subsampling layer").build();

		OutputLayer layer6 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
				.activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).name("Output").nOut(outputNum).build();

		Map<Integer, Double> lrSchedule = new HashMap<>();
		lrSchedule.put(0, 0.06); // iteration #, learning rate
		lrSchedule.put(200, 0.05);
		lrSchedule.put(600, 0.028);
		lrSchedule.put(800, 0.0060);
		lrSchedule.put(1000, 0.001);

		log.info("BUILD MODEL");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngseed) //
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //
				.updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, lrSchedule))).l2(1e-4) //
				.list() //
				.layer(0, layer0).layer(1, layer1).layer(2, layer2).layer(3, layer3).layer(4, layer4).layer(5, layer5)
				.layer(6, layer6).pretrain(false).backprop(true)
				.setInputType(InputType.convolutional(height, width, channels)).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		// The Score iteration Listener will log
		// output to show how well the network is training
		model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

		log.info("TRAIN MODEL");
		for (int i = 0; i < 2; i++) {
			model.fit(dataIter);
		}

		log.info("EVALUATE MODEL");
		recordReader.reset();

		// The model trained on the training dataset split
		// now that it has trained we evaluate against the
		// test data of images the network has not seen

		recordReader.initialize(test, new EqualizeHistTransform());
		DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		scaler.fit(testIter);
		testIter.setPreProcessor(scaler);

		/*
		 * log the order of the labels for later use In previous versions the label
		 * order was consistent, but random In current verions label order is
		 * lexicographic preserving the RecordReader Labels order is no longer needed
		 * left in for demonstration purposes
		 */
		String testlabels = recordReader.getLabels().toString();

		// Create Eval object with 10 possible classes
		Evaluation eval = new Evaluation(outputNum);

		// Evaluate the network
		while (testIter.hasNext()) {
			DataSet next = testIter.next();
			INDArray output = model.output(next.getFeatures());
			// Compare the Feature Matrix from the model
			// with the labels from the RecordReader
			eval.eval(next.getLabels(), output);
		}

		log.info("eval stats " + eval.stats());

		log.info("traininglabels: " + trainingLabels);
		log.info("testlabels: " + testlabels);
		ModelSerializer.writeModel(model, new File("mnist-model.zip"), true);
		
		RunTracker.getRuns().put("mnist", false);

	}

}
