package com.example.demo;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Date;
import java.util.UUID;

import javax.imageio.ImageIO;

import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

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

		for (int rotate = rotateMin; rotate <= rotateMax; rotate++) {
			try {

				Resource boardRes = loader.getResource("classpath:board.png");
				Resource watermarkRes = loader.getResource("classpath:watermark.png");

				BufferedImage watermarkImg = ImageIO.read(watermarkRes.getInputStream());

				BufferedImage boardImg = ImageIO.read(boardRes.getInputStream());
				boardImg = rotateImage(boardImg, rotate);

				final BufferedImage rotatedImg = boardImg;

				// create the new image, canvas size is the max. of both image sizes
				int w = Math.max(boardImg.getWidth(), watermarkImg.getWidth());
				int h = Math.max(boardImg.getHeight(), watermarkImg.getHeight());
				BufferedImage combined = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);

				int initialWidth = boardImg.getWidth() / 2;

				int basePointHeight = h / 2;
				int basePointWidth = w / 2;
				int initialHeightOffset = 80;

				// i = 580
				for (int i = 80; i <= 580; i++) {
					if (i % 2 == 0) {
						final int idx = i;
						final int deg = rotate;
						Observable<Boolean> f = Observable.fromCallable(() -> {
							writeImg(rotatedImg, watermarkImg, w, h, combined, initialWidth, basePointWidth,
									basePointHeight - idx - 1, prefix, deg, idx);
							return true;
						}).subscribeOn(Schedulers.io());
						Observable<Boolean> n = Observable.fromCallable(() -> {
							writeImg(rotatedImg, watermarkImg, w, h, combined, initialWidth, basePointWidth,
									basePointHeight - idx, prefix, deg, idx);
							return true;
						}).subscribeOn(Schedulers.io());

						Observable.zip(f, n, (first, next) -> {
							return next;
						}).blockingSubscribe();
					}

				}

			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	static final boolean test = true;

	static String getOutputdir() {
		String outputdir = "/home/salgmachine/Schreibtisch/development/workspace/dart-net/dart-net-demo-1/src/main/resources/images/train/";
		if (test) {
			outputdir = "./";
		}
		return outputdir;
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
		g.drawImage(overlay, x, y, null);
		String fname = prefix + "_" + uuid + ".png";
		log.debug(" Drawing overlay at x=" + x + " y=" + y + " at rotate=" + rotate + " for " + fname);

		combined = rotateImage(combined, 360 - rotate);
		Path filepath = Paths.get(outPath.toFile().getAbsolutePath(), fname);
		publisher.publishEvent(new ImgWrittenEvent(filepath, new Date(), isTestImg, prefix));

		final BufferedImage finalImg = resizeImageAndGrayscale(combined);
		ImageIO.write(finalImg, "PNG", filepath.toFile());
	}

	private BufferedImage resizeImageAndGrayscale(BufferedImage img) {
		return Scalr.resize(img, Method.ULTRA_QUALITY, img.getWidth() / 4, img.getHeight() / 4, Scalr.OP_GRAYSCALE);
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
