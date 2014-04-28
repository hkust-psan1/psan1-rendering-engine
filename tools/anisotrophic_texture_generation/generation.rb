require 'pnm'
require 'chunky_png'

RGB_SCALE = 255

def len(vec)
	return Math::sqrt(vec[0] * vec[0] + vec[1] * vec[1])
end

def normalize(vec)
	length = len(vec)
	return [vec[0] / length * RGB_SCALE, vec[1] / length * RGB_SCALE]
end

image_height = 1024
image_width = 1024

center_x = image_width / 2
center_y = image_height / 2

height_ratio = 255.0 / (image_height / 2)
width_ratio = 255.0 / (image_width / 2)

image_data = []

(0 .. image_height).each do |i|
	image_row = []
	(0 .. image_width).each do |j|
		# for generating elavator door texture
		rand_1 = rand(2) - 1
		rand_2 = rand(2) - 1
		image_row << [127 + rand_1, 127 + rand_2, 255]

		"""
		x_diff = 1.0 * j - center_x
		y_diff = 1.0 * i - center_y

		x_dir = x_diff
		y_dir = y_diff

		diff_length = len([x_diff, y_diff])

		half_width = image_width / 2
		half_height = image_height / 2
		half_diagnol = Math::sqrt(half_width * half_width + half_height * half_height)

		red = x_dir / image_width * 255 + 128
		green = y_dir / image_height * 255 + 128
		blue = diff_length / half_diagnol * 127

		pixel_value = [red.to_i, green.to_i, blue.to_i]
		image_row << pixel_value
		"""
	end

	image_data << image_row
end

output = PNM::Image.new(image_data, :type => :ppm)

ppm_name = 'normal.ppm'

output.write(ppm_name)


"""
pngs = Dir.entries('./inputs').select do |f|
	f.match(/png$/)
end

pngs.each do |png_name|
	image = ChunkyPNG::Image.from_file('inputs/' + png_name)

	image_data = []

	(0 .. image.dimension.height-1).each do |i|
		image_row = []
		(0 .. image.dimension.width-1).each do |j|
			red = ChunkyPNG::Color.r(image[j,i])
			green = ChunkyPNG::Color.g(image[j,i])
			blue = ChunkyPNG::Color.b(image[j,i])

			image_row << [red, green, blue]
		end

		image_data << image_row
	end

	output = PNM::Image.new(image_data, :type => :ppm)

	ppm_name = png_name[0 .. png_name.length - 4] + 'ppm'
	puts ppm_name

	output.write('outputs/' + ppm_name)
end
"""