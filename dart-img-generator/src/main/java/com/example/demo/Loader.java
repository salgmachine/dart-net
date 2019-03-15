package com.example.demo;

import static marvinplugins.MarvinPluginCollection.blackAndWhite;
import static marvinplugins.MarvinPluginCollection.halftoneDithering;
import static marvinplugins.MarvinPluginCollection.histogramEqualization;
import static marvinplugins.MarvinPluginCollection.invertColors;
import static marvinplugins.MarvinPluginCollection.morphologicalDilation;
import static marvinplugins.MarvinPluginCollection.morphologicalErosion;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
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
import marvin.image.MarvinImage;
import marvin.io.MarvinImageIO;
import marvin.util.MarvinImageUtils;

@Slf4j
@Component
public class Loader {

	@Autowired
	private ResourceLoader loader;

	@Autowired
	private ApplicationEventPublisher publisher;

	public void addWatermark(String prefix, int rotateMin, int rotateMax) throws IOException {

		Resource watermarkRes = loader.getResource("classpath:dart3.png");
		BufferedImage watermarkImg = ImageIO.read(watermarkRes.getInputStream());

		List<String> asList = Arrays.asList("classpath:board3.png", "classpath:board-5.png", "classpath:board6.png");

		List<Observable<Boolean>> collect = IntStream.range(rotateMin, rotateMax + 1).boxed().map(rot -> {
			return Observable.fromCallable(() -> {
				try {

					// watermarkImg = Scalr.resize(watermarkImg, Method.AUTOMATIC, 64, 64);

					final BufferedImage watermark = watermarkImg;

					String boardImgPath = asList.get(new Random().nextInt(3));

					Resource boardRes = loader.getResource(boardImgPath);

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

					int start = 180; // 180
					int max = 680; // 680
					int stepsize = 3;

					for (int i = start; i <= max;) {
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

		if (idx % 2 == 0) {
			// overlay = rotateImage(overlay, 180);
		}

		g.drawImage(overlay, drawOverlayAtX, drawOverlayAtY, null);
		String fname = prefix + "_(x" + x + " y" + y + " rot" + rotate + ")_" + UUID.randomUUID().toString() + ".png";

		// log.info(" Drawing overlay at x=" + x + " y=" + y + " at rotate=" + rotate +
		// " for " + fname);

		combined = rotateImage(combined, 360 - rotate);

		Path filepath = Paths.get(outPath.toFile().getAbsolutePath(), fname);
		publisher.publishEvent(new ImgWrittenEvent(filepath, new Date(), isTestImg, prefix));

		BufferedImage finalImg = postProcess(combined);

		BufferedImage copy = new BufferedImage(finalImg.getWidth(), finalImg.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = copy.createGraphics();
		g2d.setColor(Color.WHITE); // Or what ever fill color you want...
		g2d.fillRect(0, 0, copy.getWidth(), copy.getHeight());
		g2d.drawImage(finalImg, 0, 0, null);
		g2d.dispose();

		ImageIO.write(finalImg, "PNG", filepath.toFile());
		transformImg(finalImg, filepath);
	}

	private void transformImg(BufferedImage img, Path path) {

		String[] fpath = path.toFile().getAbsolutePath().split("\\.");

		MarvinImage originalImage = new MarvinImage(img, "PNG");
		MarvinImage image = originalImage.clone();

		blackAndWhite(image, 50);
		MarvinImageIO.saveImage(image, fpath[0] + "_bw." + fpath[1]);

		image = originalImage.clone();
		halftoneDithering(originalImage, image);
		MarvinImageIO.saveImage(image, fpath[0] + "_dt." + fpath[1]);

		image = originalImage.clone();
		morphologicalDilation(originalImage, image, new boolean[][] { { true, true }, { true, false } });
		MarvinImageIO.saveImage(image, fpath[0] + "_dl." + fpath[1]);

		image = originalImage.clone();
		morphologicalErosion(originalImage, image, new boolean[][] { { true, true }, { true, false } });
		MarvinImageIO.saveImage(image, fpath[0] + "_er." + fpath[1]);

		image = originalImage.clone();
		invertColors(originalImage, image);
		MarvinImageIO.saveImage(image, fpath[0] + "_ic." + fpath[1]);

	}

	private BufferedImage postProcess(BufferedImage img) {
		int rnd = new Random().nextInt(340);
		BufferedImage result = Scalr.resize(img, Method.SPEED, 448, 448);

//		if(rnd % 3 == 0) {
//			result = rotateImage(result, rnd);
//		} 
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
