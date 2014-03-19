require 'pnm'
require 'chunky_png'

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