package com.example.demo;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ColorConvertOp;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.imageio.ImageIO;

import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.imgscalr.Scalr.Rotation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.core.env.Environment;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

import com.google.common.collect.Lists;

import io.reactivex.Observable;
import io.reactivex.schedulers.Schedulers;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Component
public class Loader {

	@Autowired
	private ResourceLoader loader;

	@Autowired
	private ApplicationEventPublisher publisher;

	public void addWatermark(String prefix, int rotateMin, int rotateMax) {

		List<Observable<Boolean>> collect = IntStream.range(rotateMin, rotateMax + 1).boxed().map(rot -> {
			return Observable.fromCallable(() -> {
				try {

					Resource boardRes = loader.getResource("classpath:board3.png");
					Resource watermarkRes = loader.getResource("classpath:dart3.png");

					BufferedImage watermarkImg = ImageIO.read(watermarkRes.getInputStream());
					// watermarkImg = Scalr.resize(watermarkImg, Method.AUTOMATIC, 64, 64);

					final BufferedImage watermark = watermarkImg;

					BufferedImage boardImg = ImageIO.read(boardRes.getInputStream());
					boardImg = rotateImage(boardImg, rot);

					final BufferedImage rotatedImg = boardImg;

					// create the new image, canvas size is the max. of both image sizes
					int w = Math.max(boardImg.getWidth(), watermarkImg.getWidth());
					int h = Math.max(boardImg.getHeight(), watermarkImg.getHeight());
					BufferedImage combined = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);

					int initialWidth = boardImg.getWidth() / 2;

					int basePointHeight = h / 2;
					int basePointWidth = w / 2;
					int initialHeightOffset = 80;

//					for (int i = 180; i <= 680; i++) {

					final int r = rot;

					int start = 180;
					int max = 680;
					int stepsize = 20;
					
					for (int i = start; i <= max; ) {
						i = i + stepsize;
						
						final int idx = i;
						Observable<Boolean> f = Observable.fromCallable(() -> {
							writeImg(rotatedImg, watermark, w, h, combined, initialWidth, basePointWidth, idx, prefix,
									r, idx);
							return true;
						}).subscribeOn(Schedulers.io());
						f.blockingFirst();
					}

				} catch (IOException e) {
					e.printStackTrace();
				}
				return true;
			}).subscribeOn(Schedulers.io());
		}).collect(Collectors.toList());

		Lists.partition(collect, 4).forEach(obs -> Observable.zip(obs, n -> {
			return true;
		}).blockingFirst());

	}

	@Autowired
	private Environment env;

	private String getOutputdir() {
		return env.getProperty("dartnet.output", String.class);
	}

	private void writeImg(BufferedImage image, BufferedImage overlay, int w, int h, BufferedImage combined,
			int initialWidth, Integer x, Integer y, String prefix, int rotate, int idx) throws IOException {
		String uuid = UUID.randomUUID().toString();

		Path outPath;

		boolean isTestImg = (idx % 8 == 0);

		Path p = Paths.get(getOutputdir(), "dart-net", isTestImg ? "test" : "train", prefix);
		if (!Files.exists(p)) {
			outPath = Files.createDirectories(p);
		} else {
			outPath = p;
		}

		Graphics g = combined.getGraphics();
		g.drawImage(image, 0, 0, null);
		
		
		int drawOverlayAtX = x - (overlay.getWidth() / 2);
		int drawOverlayAtY = y - overlay.getHeight();
		
		int rgb = image.getRGB(drawOverlayAtX, drawOverlayAtY);
		
		g.drawImage(overlay, drawOverlayAtX, drawOverlayAtY, null);
		String fname = prefix + "_(x" + x + " y" + y + " rot" + rotate + " color"+rgb+")_" + UUID.randomUUID().toString() + ".png";

		log.info(" Drawing overlay at x=" + x + " y=" + y + " at rotate=" + rotate + " for " + fname);

		combined = rotateImage(combined, 360 - rotate);
		Path filepath = Paths.get(outPath.toFile().getAbsolutePath(), fname);
		publisher.publishEvent(new ImgWrittenEvent(filepath, new Date(), isTestImg, prefix));

		final BufferedImage finalImg = resizeImageAndGrayscale(combined);
		ImageIO.write(finalImg, "PNG", filepath.toFile());
	}

	
	private BufferedImage resizeImageAndGrayscale(BufferedImage img) {
		int rnd = new Random().nextInt(340);
		BufferedImage result = Scalr.resize(img, Method.AUTOMATIC, 448, 448); 
		if(rnd % 3 == 0) {
			result = rotateImage(result, rnd);
		}
		if(rnd % 7 == 0) {
			result = Scalr.rotate(result, Rotation.FLIP_HORZ);
		}
		return result;
	}

	private BufferedImage rotateImage(BufferedImage sourceImage, double angle) {
		int width = sourceImage.getWidth();
		int height = sourceImage.getHeight();
		BufferedImage destImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2d = destImage.createGraphics();

		AffineTransform transform = new AffineTransform();
		transform.rotate(angle / 180 * Math.PI, width / 2, height / 2);
		g2d.drawRenderedImage(sourceImage, transform);

		g2d.dispose();
		return destImage;
	}
}
