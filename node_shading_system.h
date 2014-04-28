#ifndef NODE_SHADING_SYSTEM_
#define NODE_SHADING_SYSTEM_

#include <assert.h>
#include <optixu/optixu_vector_types.h>

/*
struct Node;
struct ShaderNode;
struct MixShaderNode;
struct DiffuseShaderNode;
struct GlossyShaderNode;
struct GlassShaderNode;
struct SubsurfScatteringShaderNode;
struct TextureNode;
struct MultiplyNode;
*/

struct ShadingCoeff {
	ShadingCoeff() {
		diffuse_color = make_float3(0);
		has_diffuse_map = false;
		diffuse_map_filename = NULL;

		has_normal_map = false;
		normal_map_factor = 1;
		normal_map_filename = NULL;
	}

	float3 diffuse_color;
	bool has_diffuse_map;
	char* diffuse_map_filename;

	float3 glossy_color;
	float glossy_roughness;

	float3 glass_color;
	float glass_roughness;

	float3 scatter_color;
	float scatter_dropoff_rate;

	bool has_normal_map;
	float normal_map_factor;
	char* normal_map_filename;

	bool has_specular_map;
	char* specular_map_filename;
};

struct Node {
	virtual ~Node();
};

struct OutputNode : public Node {
	Node* color_input;
	Node* normal_input;
};

struct TextureNode : public Node {
	TextureNode() {
		texture_filename = NULL;
		multiplier = 1;
	}

	char* texture_filename;
	float multiplier;
};

struct ShaderNode : public Node {
	Node* output;
};

struct MixShaderNode : public ShaderNode {
	MixShaderNode() {
		input1 = NULL;
		input2 = NULL;
		fac_node = NULL;
		fac = 0.5;
	}

	ShaderNode* input1;
	ShaderNode* input2;

	// when fac_node is not provided, use fac number
	Node* fac_node;
	float fac;
};

struct DiffuseShaderNode : public ShaderNode {
	DiffuseShaderNode() {
		texture_input = NULL;
		diffuse_color = make_float3(1);
	}

	TextureNode* texture_input;
	float3 diffuse_color;
};

struct GlossyShaderNode : public ShaderNode {
	GlossyShaderNode() {
		refl_color = make_float3(1);
		roughness = 0;
		has_anisotrophic_map = false;
		anisotrophic_map_filename = NULL;
	}

	float3 refl_color;
	float roughness;

	bool has_anisotrophic_map;
	char* anisotrophic_map_filename;
};

struct GlassShaderNode : public ShaderNode {
	GlassShaderNode() {
		refl_color = make_float3(1);
		refr_color = make_float3(1);
		roughness = 0;
	}

	float3 refl_color;
	float3 refr_color;
	float roughness;
};

struct SubsurfScatteringShaderNode : public ShaderNode {
	SubsurfScatteringShaderNode() {
		scatter_color = make_float3(1);
		dropoff_rate = 0.2;
	}

	float3 scatter_color;
	float dropoff_rate;
};

void process_shader_node(ShaderNode*, ShadingCoeff&, float);

void process_mix_shader_node(MixShaderNode* mix_shader_node, ShadingCoeff& sc, float coeff = 1) {
	Node* fac_node = mix_shader_node->fac_node;
	if (fac_node) {
		TextureNode* fac_texture = dynamic_cast<TextureNode*>(fac_node);
		sc.has_specular_map = true;
		sc.specular_map_filename = fac_texture->texture_filename;
	}

	float fac = mix_shader_node->fac;
	process_shader_node(mix_shader_node->input1, sc, fac);
	process_shader_node(mix_shader_node->input2, sc, 1 - fac);
}

void process_shader_node(ShaderNode* shader_node, ShadingCoeff& sc, float coeff = 1) {
	MixShaderNode* mix_shader = dynamic_cast<MixShaderNode*>(shader_node);
	if (mix_shader) {
		process_mix_shader_node(mix_shader, sc, coeff);
	} else { // not mix shader
		DiffuseShaderNode* diffuse_shader = dynamic_cast<DiffuseShaderNode*>(shader_node);
		GlossyShaderNode* glossy_shader = dynamic_cast<GlossyShaderNode*>(shader_node);

		if (diffuse_shader) {
			if (diffuse_shader->texture_input) {
				sc.diffuse_map_filename = diffuse_shader->texture_input->texture_filename;
				sc.has_diffuse_map = true;
			} else { // no diffuse texture
				sc.diffuse_color = coeff * diffuse_shader->diffuse_color;
			}
		} else if (glossy_shader) {
			sc.glossy_color = coeff * glossy_shader->refl_color;
			sc.glass_roughness = glossy_shader->roughness;
		}
	}
}

ShadingCoeff generate_shading_coeff_from_output_node(OutputNode* output_node) {
	ShadingCoeff sc;

	// color
	ShaderNode* shader_node = dynamic_cast<ShaderNode*>(output_node->color_input);
	if (shader_node) {
		process_shader_node(shader_node, sc);
	}

	// normal
	Node* normal_input = output_node->normal_input;
	if (normal_input) {
		TextureNode* normal_input_texture = dynamic_cast<TextureNode*>(normal_input);

		assert(normal_input_texture);

		sc.normal_map_factor = normal_input_texture->multiplier;
		sc.normal_map_filename = normal_input_texture->texture_filename;
	}

	return sc;
}

#endif