import { Jimp } from "jimp";

export async function save3DBinaryArrayToPNG(
  data: number[][][],
  outputName: string,
): Promise<void> {
  // 1. Автоматически определяем высоту и ширину
  const height = 200;
  const width = 200;

  // if (height === 0 || width === 0) {
  //   console.error("❌ Массив пустой или неправильной структуры");
  //   return;
  // }

  console.log(
    `📐 Размер изображения: ${width} x ${height} (всего ${width * height} пикселей)`,
  );

  // 2. Создаём картинку
  const image = new Jimp({ width, height });

  // 3. Заполняем пиксели
  for (let y = 0; y < height; y++) {
    const row = data[y];
    for (let x = 0; x < width; x++) {
      // Берём значение из data[y][x][0] (если нет данных — ставим 0)
      const value = row?.[x]?.[0] ?? 0;

      // 0 -> чёрный (0xFF000000), 1 -> белый (0xFFFFFFFF)
      const color = value === 1 ? 0xffffffff : 0xff000000;

      image.setPixelColor(color, x, y);
    }
  }

  await image.write(`./out/${outputName}`);
}
