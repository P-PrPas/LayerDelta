// Rewrite PNG palette entries to their indices so the browser decoder retains the truth class.
export function indexPalette(png: Uint8Array<ArrayBuffer>) {
  if (png[0] !== 137 || png[1] !== 80 || png[2] !== 78 || png[3] !== 71 || png.length < 33) return png;
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const chunks = [png.subarray(0, 8)];
  for (let at = 8; at + 12 <= png.length;) {
    const size = view.getUint32(at);
    const end = at + size + 12;
    if (end > png.length) return png;
    const type = view.getUint32(at + 4);
    if (type === 0x504c5445 && png[25] === 3 && size % 3 === 0 && size <= 768) {
      const chunk = png.slice(at, end);
      for (let i = 0; i < size; i++) chunk[8 + i] = Math.floor(i / 3);
      let crc = -1;
      for (const byte of chunk.subarray(4, 8 + size)) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
      }
      new DataView(chunk.buffer).setUint32(8 + size, ~crc);
      chunks.push(chunk);
    } else if (type !== 0x74524e53 && type !== 0x67414d41 && type !== 0x69434350 && type !== 0x73524742) {
      chunks.push(png.subarray(at, end));
    }
    at = end;
  }
  const result = new Uint8Array(chunks.reduce((length, chunk) => length + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {result.set(chunk, offset); offset += chunk.length;}
  return result;
}

const palette = [[95, 220, 180], [248, 173, 97], [182, 150, 246], [244, 114, 150],
  [112, 181, 248], [247, 223, 104], [130, 219, 128], [220, 172, 114]];

export async function paintMask(canvas: HTMLCanvasElement, bytes: ArrayBuffer, type: string, ignore: number, signal: AbortSignal) {
  const bitmap = await createImageBitmap(new Blob([indexPalette(new Uint8Array(bytes))], {type}), {colorSpaceConversion: "none"});
  try {
    if (signal.aborted) return;
    if (bitmap.width * bitmap.height > 16_000_000) throw new Error("Truth mask is too large to display.");
    const context = canvas.getContext("2d", {willReadFrequently: true});
    if (!context) throw new Error("Truth mask is unavailable.");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    context.drawImage(bitmap, 0, 0);
    const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
    for (let i = 0; i < image.data.length; i += 4) {
      const index = image.data[i];
      image.data.set(palette[index % palette.length], i);
      image.data[i + 3] = index === ignore ? 0 : 115;
    }
    context.putImageData(image, 0, 0);
  } finally {bitmap.close();}
}
