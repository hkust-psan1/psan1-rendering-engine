
input_obj = File.new 'court.obj', 'r'

input_obj_data = []

objs = []
curr_obj = nil

while line = input_obj.gets
	if line[0] == 'o'
		obj_name = line.split(' ')[1]

		objs << curr_obj if curr_obj != nil # do no add unless initialized
		curr_obj = []
	end

	if curr_obj != nil # curr_obj already initialized
		curr_obj << line
	end
end

objs.each do |obj|
	output_filename = 'outputs/' + obj[0].split(' ')[1] + '.obj'
	File.open(output_filename, 'w') do |f|
		f.write obj.join('')
	end
end
