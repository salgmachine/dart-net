package com.example.demo;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;

import com.google.common.collect.Lists;

import io.reactivex.Observable;
import io.reactivex.ObservableSource;
import io.reactivex.Scheduler;
import io.reactivex.schedulers.Schedulers;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@SpringBootApplication
public class DartImgGeneratorApplication {

	// degree offset for next number: 2 degrees
	// degree offset for left to right: 16 degrees
	public static final List<Run> runs = Arrays.asList(new Run("20", -8, -8 + 16), new Run("3", 10, 10 + 16),
			new Run("16", 28, 28 + 16), new Run("7", 46, 46 + 16), new Run("12", 64, 64 + 16),
			new Run("11", 82, 82 + 16), new Run("8", 100, 100 + 16), new Run("15", 118, 118 + 16),
			new Run("4", 136, 136 + 16), new Run("19", 154, 154 + 16), new Run("2", 172, 172 + 16),
			new Run("17", 190, 190 + 16), new Run("6", 208, 208 + 16), new Run("13", 226, 226 + 16),
			new Run("10", 244, 244 + 16), new Run("9", 262, 262 + 16), new Run("14", 280, 280 + 16),
			new Run("5", 298, 298 + 16), new Run("18", 314, 314 + 16), new Run("1", 332, 332 + 16)
//			
	);

	public static void main(String[] args) {
		ConfigurableApplicationContext ctx = SpringApplication.run(DartImgGeneratorApplication.class, args);
		Loader bean = ctx.getBean(Loader.class);

		List<Observable<Run>> observableRuns = runs.stream().map(run -> {
			return Observable.fromCallable(() -> {
				return run.execute(bean);
			}).subscribeOn(Schedulers.newThread());
		}).collect(Collectors.toList());

		Lists.partition(observableRuns, 8).forEach(batch -> Observable.zip(batch, (n) -> Boolean.TRUE).blockingFirst());

//		runs.forEach(run -> {
//			bean.addWatermark(run.getPrefix(), run.getRotateMin(), run.getRotateMax());
//		});

	}

//	public static BufferedImage createResizedCopy(Image originalImage, int scaledWidth, int scaledHeight,
//			boolean preserveAlpha) {
//		System.out.println("resizing...");
//		int imageType = preserveAlpha ? BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB;
//		BufferedImage scaledBI = new BufferedImage(scaledWidth, scaledHeight, imageType);
//		Graphics2D g = scaledBI.createGraphics();
//		if (preserveAlpha) {
//			g.setComposite(AlphaComposite.Src);
//		}
//		g.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight, null);
//		g.dispose();
//		return scaledBI;
//	}

//	public static BufferedImage rotateImageByDegrees(BufferedImage img, double angle) {
//
//	    Graphics2D g = img.createGraphics();
//	    g.translate(img.getWidth() / 2, img.getHeight() / 2);
//	    g.rotate(Math.toRadians(angle));
////	    g.translate(img.getWidth(), img.getHeight());
//	    
//	    
//	    //-----------------------MODIFIED--------------------------------------
//	    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON) ;
//	    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC) ;
//	    g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY) ;
//
//	    g.drawImage(img, 0, 0, img.getWidth(), img.getHeight(), null);
//	    
//	    return img;
//	}

	class Ranges {
		List<Integer> twentyX = Arrays.asList(-14, -15, -16, -17);
		List<Integer> twentyY = Arrays.asList(70);
	}

}
