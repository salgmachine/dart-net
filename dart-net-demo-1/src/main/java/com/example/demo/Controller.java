package com.example.demo;

import java.util.HashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.core.task.TaskExecutor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.nets.AlexNet;
import com.example.demo.nets.ConvNet;
import com.example.demo.nets.MnistNet;
import com.example.demo.nets.SimpleCNN;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@RestController
public class Controller {

	@Autowired
	private Environment env;

	@Autowired
	private TaskExecutor executor;

	@GetMapping("/simple")
	public ResponseEntity<?> simple() throws Exception {
		String inputpath = env.getProperty("dartnet.input", String.class);
		log.info("REST Request to run SimpleCNN");
		log.info("Using input base path {}", inputpath);

		executor.execute(() -> {
			try {
				SimpleCNN c = new SimpleCNN();
				c.run(inputpath);
				RunTracker.getRuns().put("SimpleCNN", false);
			} catch (Exception e) {
				log.error("Error running alexnet: {}", e);
				RunTracker.getRuns().put("SimpleCNN", false);
			}
		});

		return ResponseEntity.ok().build();
	}

	@GetMapping("/alexnet")
	public ResponseEntity<?> alexnet() throws Exception {
		String inputpath = env.getProperty("dartnet.input", String.class);
		log.info("REST Request to run alexnet");
		log.info("Using input base path {}", inputpath);

		executor.execute(() -> {
			try {
				AlexNet alexNet = new AlexNet();
				alexNet.run(inputpath);
				RunTracker.getRuns().put("alexnet", false);
			} catch (Exception e) {
				log.error("Error running alexnet: {}", e);
				RunTracker.getRuns().put("alexnet", false);
			}
		});

		return ResponseEntity.ok().build();
	}

	@GetMapping("/deep")
	public ResponseEntity<?> deepConv() {
		String inputpath = env.getProperty("dartnet.input", String.class);
		log.info("REST Request to run deep convnet");
		log.info("Using input base path {}", inputpath);

		executor.execute(() -> {
			try {
				ConvNet net = new ConvNet();
				net.run(inputpath);
				RunTracker.getRuns().put("deep", false);
			} catch (Exception e) {
				log.error("Error running deep convnet: {}", e);
				RunTracker.getRuns().put("deep", false);
			}

		});
		return ResponseEntity.ok().build();
	}

	@GetMapping("/mnist")
	public ResponseEntity<?> mnist() {
		String inputpath = env.getProperty("dartnet.input", String.class);
		log.info("REST Request to run deep convnet");
		log.info("Using input base path {}", inputpath);

		executor.execute(() -> {
			try {
				MnistNet net = new MnistNet();
				net.run(inputpath);
				RunTracker.getRuns().put("mnist", false);
			} catch (Exception e) {
				log.error("Error running deep convnet: {}", e);
				RunTracker.getRuns().put("mnist", false);
			}

		});
		return ResponseEntity.ok().build();
	}
}
