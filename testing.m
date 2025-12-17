input_size = 3
output_size = 3
hidden_layer_sizes = [5 5]
input = [2,2,2]

f = eye(input_size+1)
sizes = [input_size hidden_layer_sizes output_size]
for i= 1:length(sizes)-1
    input_length = sizes(i);
    output_length = sizes(i+1);

    weight = ones(input_length,output_length);
    projective_column = zeros(input_length,1);
    translational_row = zeros(1,output_length);
    layer = [weight projective_column;translational_row 1];
    f = f*layer
end
f
[input 1]*f