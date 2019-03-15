package com.example.demo.nets.web;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.Darknet19;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.deeplearning4j.zoo.model.Xception;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@RestController
@RequestMapping("/nets")
public class TestNetsApi {

	@Autowired
	private Environment env;

	@GetMapping("/googlenet")
	public void testgoogleNet() throws Exception {

		int[][] shape = new int[3][3];
		shape[0] = new int[] { 3, 750, 750 };

		// runs into NAN
		GoogLeNet lnet = new GoogLeNet(20, 123, WorkspaceMode.ENABLED);
		lnet.setInputShape(shape);
		ComputationGraph init = new ComputationGraph(lnet.conf());
		runComputationGraph(init, 10000, "googlenet");
	}

	@GetMapping("/vgg16")
	public void testVGG16() throws Exception {
		// runs into NAN
		VGG16 build = org.deeplearning4j.zoo.model.VGG16.builder().seed(123).inputShape(new int[] { 3, 750, 750 })
				.cudnnAlgoMode(AlgoMode.PREFER_FASTEST).numClasses(20).build();
		ComputationGraph init = build.init();
		runComputationGraph(init, 10000, "vgg16");
	}

	@GetMapping("/vgg19")
	public void testVGG19() throws Exception {
		// runs into NAN
		VGG19 build = org.deeplearning4j.zoo.model.VGG19.builder().seed(123).inputShape(new int[] { 3, 480, 480 })
				.cudnnAlgoMode(AlgoMode.PREFER_FASTEST).numClasses(20).build();
		// Model mm = build.initPretrained(PretrainedType.IMAGENET);
		// ComputationGraph init = (ComputationGraph) mm;
		ComputationGraph init = build.init();
		runComputationGraph(init, 10000, "vgg19");

	}

	@GetMapping("/xceptionnet")
	public void testXceptionNet() throws Exception {
		// 121 layer oO .. produces high load even on single batches
		Xception build = org.deeplearning4j.zoo.model.Xception.builder().seed(123).inputShape(new int[] { 3, 480, 480 })
				.cudnnAlgoMode(AlgoMode.PREFER_FASTEST).numClasses(20).build();
		ComputationGraph init = build.init();
		runComputationGraph(init, 10000, "xceptionnet");
	}

	@GetMapping("/squeezenet")
	public void testSqueezeNet() throws Exception {
		SqueezeNet build = org.deeplearning4j.zoo.model.SqueezeNet.builder().seed(123)
				.cudnnAlgoMode(AlgoMode.PREFER_FASTEST).inputShape(new int[] { 3, 750, 750 }).numClasses(20).build();
		ComputationGraph init = build.init();
		runComputationGraph(init, 10000, "squeezenet");
	}

	@GetMapping("/lenet")
	public void testLeNet() throws Exception {
		// converges pretty fast
		LeNet build = org.deeplearning4j.zoo.model.LeNet.builder().seed(123).inputShape(new int[] { 3, 750, 750 })
				.cudnnAlgoMode(AlgoMode.PREFER_FASTEST).numClasses(20).build();
		MultiLayerConfiguration net = build.conf();
		runMultiLayerConfiguration(net, 10000);
	}

	private ComputationGraph restoredDarknet;

	public ComputationGraph getRestoredDarknet() {

		if (restoredDarknet == null) {
			Path path = getPersistedModel("darknet");
			try {
				this.restoredDarknet = ModelSerializer.restoreComputationGraph(path.toFile());
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		return restoredDarknet;
	}

	@PostMapping("/darknet")
	public ResponseEntity<?> uploadTestImg(@RequestParam("file") MultipartFile file) {
		log.info("received file {}", file);

		ComputationGraph restoreComputationGraph = getRestoredDarknet();

		if (restoreComputationGraph == null) {
			return ResponseEntity.notFound().build();
		}

		try {

			String name = file.getOriginalFilename();

			String ft = name.substring(0, 4).split("_")[0];
			List<String> labels = new ArrayList<>();
			labels.add(ft);

			Float score = evalImage(file, restoreComputationGraph, labels);
			Map<String, Object> response = new HashMap<>();
			response.put("score", score);
			response.put("label", ft);
			response.put("filename", name);

			return ResponseEntity.ok(response);

		} catch (Exception e) {
			return ResponseEntity.status(500).body(e);
		}

	}

	private Float evalImage(MultipartFile imageFile, ComputationGraph model, List<String> labels) throws IOException {
		NativeImageLoader loader = new NativeImageLoader(448, 448, 3);
		INDArray image = loader.asMatrix(convert(imageFile));
		ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
		preProcessor.transform(image);
		INDArray output = model.outputSingle(false, image);

		return output.getFloat(0);
	}

	private File convert(MultipartFile file) throws IOException {
		File convFile = new File(file.getOriginalFilename());
		convFile.createNewFile();
		FileOutputStream fos = new FileOutputStream(convFile);
		fos.write(file.getBytes());
		fos.close();
		return convFile;
	}

	@GetMapping("/darknet")
	public void testDarknet() throws Exception {

		String label = "darknet";

		ComputationGraph mdl = getRestoredDarknet();
		if (mdl == null) {
			Darknet19 build = org.deeplearning4j.zoo.model.Darknet19.builder().seed(123)
					.cudnnAlgoMode(AlgoMode.PREFER_FASTEST).inputShape(new int[] { 3, 448, 448 }).numClasses(20)
					.build();
			 mdl = build.init();
			 
//			mdl = (ComputationGraph) build.initPretrained();

//			FineTuneConfiguration ft = new FineTuneConfiguration.Builder()
//					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123).build();
//			ComputationGraph newModel = new TransferLearning.GraphBuilder(mdl).fineTuneConfiguration(ft)
//					.removeVertexAndConnections("softmax")
//					.removeVertexAndConnections("loss")
//					.addLayer("output",
//							new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(20).nIn(1000)
//									.weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build(),
//							"globalpooling").setOutputs("output")
//					.build();
//			mdl = newModel;
		} else {
			log.info("Restoring darknet model ..");
		}

		runComputationGraph(mdl, 10000, label);
	}

	private void runComputationGraph(ComputationGraph graph, int batches, String label) throws Exception {
		RunProvisioner p = new RunProvisioner(env.getProperty("dartnet.input", String.class), 448, 448, 3, 2, 20)
				.withTerminateAfterBatches(batches);
		graph = p.setup(graph, label);
		graph.fit(p.getDataIterator(), 2);
		p.evaluateCg(graph, label);
	}

	private void runMultiLayerConfiguration(MultiLayerConfiguration model, int batches) throws Exception {

		RunProvisioner p = new RunProvisioner(env.getProperty("dartnet.input"), 750, 750, 3, 2, 20)
				.withTerminateAfterBatches(batches);
		MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(model);
		multiLayerNetwork = p.setup(multiLayerNetwork);
		multiLayerNetwork.fit(p.getDataIterator());
		p.evaluateMln(multiLayerNetwork);
	}

	private class RunProvisioner {

		private final Logger log = LoggerFactory.getLogger(RunProvisioner.class);
		private final String basepath;
		private final int height;
		private final int width;
		private final int channels;
		private final int batchSize;
		private final int classes;
		private Integer terminateAfter;

		private DataSetIterator dataIterator;
		private String trainingLabels;
		private FileSplit testFilesplit;
		private ImageRecordReader recordReader;
		private ImageRecordReader testRecordReader;
		private DataSetIterator testIter;

		private StatsStorage statsStorage;

		private void startUiServer() {
			// Initialize the user interface backend
			UIServer uiServer = UIServer.getInstance();
			// Configure where the network information (gradients, score vs. time etc) is to
			// be stored. Here: store in memory.

			Path o = Paths.get(System.getProperty("user.dir"), "stats");

			statsStorage = new FileStatsStorage(o.toFile()); // Alternative: new
																// FileStatsStorage(File),
																// for
			// saving
			// and loading later
			// Attach the StatsStorage instance to the UI: this allows the contents of the
			// StatsStorage to be visualized
			uiServer.attach(statsStorage);

		}

		public RunProvisioner withTerminateAfterBatches(int batches) {
			this.terminateAfter = batches;
			return this;
		}

		public StatsStorage getStatsStorage() {
			return statsStorage;
		}

		DataSetIterator getTestDataIterator() {
			return testIter;
		}

		DataSetIterator getDataIterator() {
			return dataIterator;
		}

		public String getTrainingLabels() {
			return trainingLabels;
		}

		public RunProvisioner(String basepath, int height, int width, int channels, int batchSize, int classes) {
			super();
			this.basepath = basepath;
			this.height = height;
			this.width = width;
			this.channels = channels;
			this.batchSize = batchSize;
			this.classes = classes;
		}

		private Path modelDirectory() {
			Path path = Paths.get(basepath, "models");
			if (Files.exists(path) == false) {
				try {
					Files.createDirectories(path);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			return path;
		}

		private void initReaders() throws IOException {
			File trainData = new File(basepath + "/train");
			File testData = new File(basepath + "/test");

			// Define the FileSplit(PATH, ALLOWED FORMATS,random)
			FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random());
			testFilesplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random());

			// Extract the parent path as the image label

			recordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
			testRecordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
			testRecordReader.initialize(testFilesplit);

			// Initialize the record reader
			// add a listener, to extract the name
			recordReader.initialize(train);

			// The LogRecordListener will log the path of each image read
			// used here for information purposes,
			// If the whole dataset was ingested this would place 60,000
			// lines in our logs
			// It will show up in the output with this format
			// o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png
			recordReader.setListeners(new LogRecordListener());

			// DataSet Iterator

			if (terminateAfter == null) {
				dataIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
						recordReader.getLabels().size());
				testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1,
						testRecordReader.getLabels().size());

			} else {
				dataIterator = new EarlyTerminationDataSetIterator(
						new RecordReaderDataSetIterator(recordReader, batchSize, 1, recordReader.getLabels().size()),
						terminateAfter);
				testIter = new EarlyTerminationDataSetIterator(new RecordReaderDataSetIterator(testRecordReader,
						batchSize, 1, testRecordReader.getLabels().size()), terminateAfter);
			}

			// Scale pixel values to 0-1
			DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
			scaler.fit(dataIterator);
			dataIterator.setPreProcessor(scaler);

			DataNormalization testScaler = new ImagePreProcessingScaler(0, 1);
			testScaler.fit(testIter);
			testIter.setPreProcessor(testScaler);

			trainingLabels = dataIterator.getLabels().toString();
			log.info("BUILD MODEL");
		}

		private CheckpointListener checkpointListener() {
			return new CheckpointListener.Builder(modelDirectory().toFile()).keepLast(3).saveEveryNIterations(120)
					.build();
		}

		private ComputationGraph setup(ComputationGraph graph, String label) throws Exception {
			startUiServer();
			initReaders();

			Path persistedModel = getPersistedModel(label);

			if (Files.exists(persistedModel)) {
				graph = ModelSerializer.restoreComputationGraph(persistedModel.toFile());
				log.info("Loaded Model from {}", persistedModel);
			}

			graph.addListeners(new ScoreIterationListener(5), new StatsListener(getStatsStorage()),
					new EvaluativeListener(new EarlyTerminationDataSetIterator(getTestDataIterator(), 10), 1000),
					checkpointListener());

			return graph;

		}

		public MultiLayerNetwork setup(MultiLayerNetwork model) throws Exception {
			startUiServer();
			initReaders();
			model.setListeners(new ScoreIterationListener(5), new StatsListener(getStatsStorage()),
					checkpointListener());
			return model;
		}

		private Evaluation prepareEval() throws IOException {
			log.info("EVALUATE MODEL");
			testRecordReader.reset();
			testRecordReader.initialize(testFilesplit);

			/*
			 * log the order of the labels for later use In previous versions the label
			 * order was consistent, but random In current verions label order is
			 * lexicographic preserving the RecordReader Labels order is no longer needed
			 * left in for demonstration purposes
			 */
			String testlabels = testRecordReader.getLabels().toString();

			log.info("Test labels : {}", testlabels);

			// Create Eval object with 10 possible classes
			Evaluation eval = new Evaluation(classes);
			return eval;
		}

		private void evaluateCg(ComputationGraph model, String label) throws IOException {

			Evaluation eval = prepareEval();

//			// Evaluate the network
			while (testIter.hasNext()) {
				DataSet next = testIter.next();
				INDArray testFeatures = next.getFeatures();

				for (INDArray arr : model.output(testFeatures)) {
					eval.eval(next.getLabels(), arr);
				}

			}

			log.info("Eval Stats : \r\n{}", eval.stats());

			Path path = getPersistedModel(label);
			ModelSerializer.writeModel(model, path.toFile(), true);

		}

		public Path getPersistedModel(String label) {
			Path path = Paths.get(System.getProperty("user.dir"), label + "-model.zip");
			return path;
		}

		private void evaluateMln(MultiLayerNetwork model) throws IOException {

			Evaluation eval = prepareEval();

//			// Evaluate the network
			while (testIter.hasNext()) {
				DataSet next = testIter.next();
				INDArray testFeatures = next.getFeatures();
				INDArray testLabels = next.getLabels();

				INDArray output = model.output(testFeatures);
				eval.eval(testLabels, output);

			}

			log.info("Eval Stats : \r\n{}", eval.stats());
		}
	}

	public Path getPersistedModel(String label) {
		Path path = Paths.get(System.getProperty("user.dir"), label + "-model.zip");
		log.info("Looking for persisted model at {}", path);
		return path;
	}

}
