/**
 * Client-side image compression before upload.
 *
 * iPhone photos are ~6MB (HEIC/JPEG, 4032x3024). Uploading them at full
 * size stalls on WiFi and wastes bandwidth: the backend re-encodes every
 * image down to 1280px for the vision model and 1920px for Homebox storage
 * anyway. Compressing on the client first makes the phone→server upload
 * ~15x smaller with no loss in final quality.
 */

/** Compression defaults tuned for item photos */
const DEFAULT_MAX_DIMENSION = 1920;
const DEFAULT_QUALITY = 0.8;
/** Files below this size are not worth recompressing */
const DEFAULT_MIN_BYTES = 400_000;

export interface CompressImageOptions {
	/** Maximum pixels on the longest edge (default 1920) */
	maxDimension?: number;
	/** JPEG quality 0-1 (default 0.8) */
	quality?: number;
	/** Skip files smaller than this many bytes (default 400_000) */
	minBytes?: number;
}

/**
 * Decode an image file to a bitmap, tolerating formats createImageBitmap
 * cannot handle (HEIC on some browsers) via an <img> fallback.
 */
async function decodeImage(file: File): Promise<ImageBitmap | HTMLImageElement> {
	try {
		return await createImageBitmap(file);
	} catch {
		// Fall through to <img> decoding (handles HEIC on iOS Safari)
	}

	const url = URL.createObjectURL(file);
	try {
		const img = new Image();
		img.decoding = 'async';
		img.src = url;
		await img.decode();
		if (!img.naturalWidth || !img.naturalHeight) {
			throw new Error('Image decoded to zero dimensions');
		}
		return img;
	} finally {
		// Safe to revoke once decoded; drawing still works from the decoded frame
		URL.revokeObjectURL(url);
	}
}

/**
 * Compress an image file for upload: downscale to `maxDimension` on the
 * longest edge and re-encode as JPEG.
 *
 * Never throws: on any failure the original file is returned so the
 * capture flow is never blocked.
 *
 * @param file - The image File from the camera/file picker
 * @param options - Optional overrides
 * @returns A JPEG File ready for upload (or the original if compression
 *          isn't possible or wouldn't help)
 */
export async function compressImageForUpload(
	file: File,
	options: CompressImageOptions = {}
): Promise<File> {
	const {
		maxDimension = DEFAULT_MAX_DIMENSION,
		quality = DEFAULT_QUALITY,
		minBytes = DEFAULT_MIN_BYTES,
	} = options;

	// Not an image, or already small enough — upload as-is
	if (!file.type.startsWith('image/') || file.size < minBytes) {
		return file;
	}

	// Animated formats (GIF/WEBP animation) — don't flatten to a still frame
	if (file.type === 'image/gif') {
		return file;
	}

	try {
		const source = await decodeImage(file);
		const width = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
		const height = source instanceof HTMLImageElement ? source.naturalHeight : source.height;

		// Only downscale, never upscale
		const scale = Math.min(1, maxDimension / Math.max(width, height));
		if (scale >= 1 && file.type === 'image/jpeg' && file.size < minBytes * 2) {
			if ('close' in source) source.close();
			return file;
		}

		const targetWidth = Math.max(1, Math.round(width * scale));
		const targetHeight = Math.max(1, Math.round(height * scale));

		let blob: Blob | null;
		if (typeof OffscreenCanvas !== 'undefined') {
			const canvas = new OffscreenCanvas(targetWidth, targetHeight);
			const ctx = canvas.getContext('2d');
			if (!ctx) throw new Error('No 2D context');
			ctx.drawImage(source as CanvasImageSource, 0, 0, targetWidth, targetHeight);
			blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
		} else {
			const canvas = document.createElement('canvas');
			canvas.width = targetWidth;
			canvas.height = targetHeight;
			const ctx = canvas.getContext('2d');
			if (!ctx) throw new Error('No 2D context');
			ctx.drawImage(source as CanvasImageSource, 0, 0, targetWidth, targetHeight);
			blob = await new Promise<Blob | null>((resolve) =>
				canvas.toBlob((b) => resolve(b), 'image/jpeg', quality)
			);
		}

		if ('close' in source) source.close();

		if (!blob || blob.size >= file.size) {
			// Recompression didn't help — keep the original
			return file;
		}

		const baseName = file.name.replace(/\.[^.]+$/, '') || 'image';
		return new File([blob], `${baseName}.jpg`, {
			type: 'image/jpeg',
			lastModified: Date.now(),
		});
	} catch {
		// Any decode/encode failure — upload the original untouched
		return file;
	}
}

/**
 * Compress a batch of image files concurrently.
 * Order is preserved. Never rejects.
 */
export async function compressImagesForUpload(
	files: File[],
	options: CompressImageOptions = {}
): Promise<File[]> {
	return Promise.all(files.map((file) => compressImageForUpload(file, options)));
}
