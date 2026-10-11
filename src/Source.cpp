
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>


#define MAX_ITERATIONS 100
#define C_SQUARED 9e4 //in (10^6m)^2 s^-2
#define DAMPING_FACTOR 1e-3
#define PI glm::pi<double>()

//y-> t, u_t, r, v_r, p, w_p, [o, q_o]
//distace in Mm (1 unit = 10^6 m)

struct integratorCreateInfo {
	double R_s;
	double max_timestep;
	double end_time;
	double err_tolerance;
	//assume not outputting/saving and in 2D
	bool console_output = false;
	bool saving_to_file = false;
	std::string filepath = "";
	bool is3D = false;
};

struct renderCreateInfo {
	glm::vec3 cam_pos;
	glm::ivec2 screen_size;
	int screen_scale;
	cv::Mat sphere_texture;
};

int geodesic2DFunction(double t, const double y[], double f[], void* params) {

	(void)(t);
	
	double R_s = *(double *)params;

	if (y[2] <= R_s + DAMPING_FACTOR) {
		printf("Approaching event horrizon at r = %g \n", y[2]);

		f[0] = 0.0;		//u(dt) 
		f[2] = 0.0;		//v(dr)
		f[4] = 0.0;		//w(d[phi])

		f[1] = 0.0;		//du
		f[3] = 0.0;		//dv
		f[5] = 0.0;		//dw

		return GSL_EBADFUNC;
	}

	f[0] = y[1];	//dt = u_t
	f[2] = y[3];	//dr = v_r
	f[4] = y[5];	//dp = w_p

	double r_minus_R_s = y[2] - R_s;
	double R_s_over_r = R_s / (y[2] * r_minus_R_s);

	f[1] = -R_s_over_r * y[1] * y[3];	//du
	f[3] = -C_SQUARED * R_s * y[1] * y[1] * r_minus_R_s / (2 * pow(y[2], 3)) + R_s_over_r * y[3] * y[3] / 2.0 + r_minus_R_s * y[5] * y[5];		//dv
	f[5] = -2 * y[3] * y[5] / y[2];		//dw

	return GSL_SUCCESS;
}

int geodesic3DFunction(double t, const double y[], double f[], void* params) {

	(void)(t);

	double R_s = *(double*)params;
	if (y[2] <= R_s + DAMPING_FACTOR) {
		printf("Approaching event horrizon at r = %g \n", y[2]);

		f[0] = 0.0;		//u(dt) 
		f[2] = 0.0;		//v(dr)
		f[4] = 0.0;		//w(d[phi])
		f[6] = 0.0;		//w(d[theta])

		f[1] = 0.0;		//du
		f[3] = 0.0;		//dv
		f[5] = 0.0;		//dw
		f[5] = 0.0;		//dq

		return GSL_EBADFUNC;
	}

	f[0] = y[1];	//dt = u_t
	f[2] = y[3];	//dr = v_r
	f[4] = y[5];	//dp = w_p
	f[6] = y[7];	//do = q_o

	//common variables beforehand and check no divide by zero
	double r_minus_R_s = 1 - (R_s / y[2]);
	double R_s_over_r = R_s /  (pow(y[2], 2) * r_minus_R_s);
	double cos_over_sin = 0;
	if (sin(y[6]) != 0) cos_over_sin = cos(y[6]) / sin(y[6]);

	//geodesic equations
	f[1] = -R_s_over_r * y[1] * y[3];	//du
	f[3] = -C_SQUARED * R_s * y[1] * y[1] * r_minus_R_s / (2 * pow(y[2], 2)) + R_s_over_r * y[3] * y[3] / 2.0 
			+ r_minus_R_s * y[2] * (y[7] * y[7] + pow(sin(y[6]), 2) * y[5] * y[5]);		//dv
	f[5] = -2 * y[3] * y[5] / y[2] - 2 * cos_over_sin * y[7] * y[5];		//dw
	f[7] = -2 * y[3] * y[7] / y[2] + sin(y[6]) * cos(y[6]) * y[5] * y[5];	//dq

	return GSL_SUCCESS;
}

double findInitialTimeDifferential2D(double R_s, double y[], bool massless) {
	//calculate common variable
	double minus_R_s_over_r = (1 - R_s / y[2]);
	
	//calculate the value and return it(for both null and regular invarient quantity)
	double top =  C_SQUARED + (y[3] * y[3]) / minus_R_s_over_r + y[2] * y[2] * y[5] * y[5];

	if (massless) return sqrt((top - C_SQUARED) / (C_SQUARED * minus_R_s_over_r));
	return sqrt(top / (C_SQUARED * minus_R_s_over_r));
}

double findInitialTimeDifferential3D(double R_s, double y[]) {
	//calculate common variable
	double minus_R_s_over_r = (1 - R_s / y[2]);

	//calculate the value and return it
	double top =  (y[3] * y[3]) / minus_R_s_over_r + y[2] * y[2] * (y[7] * y[7] + pow(sin(y[6]), 2) * y[5] * y[5]);

	return sqrt(top / (C_SQUARED * minus_R_s_over_r));
}

double findCircularAngularSpeeed(double R_s, double r) {
	double bottom = r * r * (2 * r - 3 * R_s);

	return sqrt(R_s * C_SQUARED / bottom);
}

void findRayDirection(double y[8], glm::vec3 cam_pos, int screen_x, int screen_y, double FOV, int width, int height, double R_s) {
	double fov_rad = FOV * PI / 180.0;
	double screen_norm_x = (-1.0f + (screen_x + 0.5f) * 2.0f / width) * tan(fov_rad / 2);
	double screen_norm_y = (1.0f - (screen_y + 0.5f) * 2.0f / height) * tan(fov_rad / 2);

	glm::vec3 local_dir = glm::normalize(glm::vec3(-1, screen_norm_y, screen_norm_x));
	
	double minus_R_s_over_r = (1 - R_s / cam_pos[0]);
	
	y[0] = 0;
	y[2] = cam_pos[0];
	y[3] = local_dir[0] / sqrt(minus_R_s_over_r);
	y[4] = cam_pos[2];
	y[5] = local_dir[2] / (cam_pos[0] * sin(cam_pos[1]));
	y[6] = cam_pos[1];
	y[7] = local_dir[1] / cam_pos[0];
	

	y[1] = findInitialTimeDifferential3D(R_s, y);
	
}

//integrates for given input perameters
int individualIntegration(double y[], double timestep, integratorCreateInfo* create_info) {
	std::ofstream output_file;
	if (create_info->saving_to_file) output_file.open(create_info->filepath, std::ios_base::out);


	gsl_odeiv2_system system;
	gsl_odeiv2_step* step;
	gsl_odeiv2_control* control = gsl_odeiv2_control_y_new(create_info->err_tolerance, create_info->err_tolerance);
	gsl_odeiv2_evolve* evolve;
	if (create_info->is3D) {
		system = { geodesic3DFunction, NULL, 8, &create_info->R_s}; 
		step = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, 8);
		evolve = gsl_odeiv2_evolve_alloc(8);
	}
	else {
		system = { geodesic2DFunction, NULL, 6, &create_info->R_s };
		step = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, 6);
		evolve = gsl_odeiv2_evolve_alloc(6);
	}

	double t = 0.0;

	int i = 0;
	while (t < create_info->end_time) {

		//double t_next = (i) * t_final / MAX_ITERATIONS;

		int success = gsl_odeiv2_evolve_apply(evolve, control, step, &system, &t, create_info->end_time, &timestep, y);

		//int success = gsl_odeiv2_driver_apply_fixed_step(driver, &t, timestep, 1000, y);

		if (success == GSL_EBADFUNC) {
			return 2;
		}
		else if (success != GSL_SUCCESS) {
			printf("ERROR:: Driver returned adaptive step error, %s: %d \n", gsl_strerror(success), success);
			return 1;
		}

		if (timestep > create_info->max_timestep) {
			timestep = create_info->max_timestep;
		}

		if (create_info->saving_to_file) {
			output_file << t << " ";		//absolute time
			output_file << y[0] << " " << y[1] << " ";	//t
			output_file << y[2] << " " << y[3] << " ";	//r
			output_file << y[4] << " " << y[5] << " ";	//phi
			if(create_info->is3D) output_file << y[6] << " " << y[7] << " ";	//theta
			output_file << std::endl;
		}

		//print every 100
		if (i % 100 == 0 && create_info->console_output) {
			printf("%.5e %.5e %.5e %.5e %.5e %.5e ", y[0], y[1], y[2], y[3], y[4], y[5]);
			if (create_info->is3D) printf("%.5e %.5e ", y[6], y[7]);
			std::cout << std::endl;
		}

		i++;
	}
	gsl_odeiv2_evolve_free(evolve);
	gsl_odeiv2_control_free(control);
	gsl_odeiv2_step_free(step);

	output_file.close();
	return 0;
}

//samples texture from sphere texture and returns its color
cv::Scalar sampleTexture(const cv::Mat& texture ,double y[]) {
	//convert ray to cartesian
	double vx = y[3] * sin(y[6]) * cos(y[4]) - y[5] * y[2] * sin(y[6]) * sin(y[4]) + y[7] * y[2] * cos(y[6]) * cos(y[4]);
	double vz = y[3] * sin(y[6]) * sin(y[4]) + y[5] * y[2] * sin(y[6]) * cos(y[4]) + y[7] * y[2] * cos(y[6]) * sin(y[4]);
	double vy = y[3] * cos(y[6]) - y[7] * y[2] * sin(y[6]);

	//normalise it
	double magnitude = sqrt(vx * vx + vy * vy + vz * vz);
	vx /= magnitude;
	vy /= magnitude;
	vz /= magnitude;

	//convert to uv coords
	double u = fmod(atan2(vz, vx) + 2.0 * PI, 2.0 * PI) / (2.0 * PI);
	double v = acos(vy) / PI;

	//convert to texture coords
	int tex_y = std::min((int)(u * texture.cols), texture.cols - 1);
	int tex_x = std::min((int)(v * texture.rows), texture.rows - 1);

	//sample texture and return it
	cv::Vec3b tex_color = texture.at<cv::Vec3b>(tex_x, tex_y);
	return cv::Scalar(tex_color[0], tex_color[1], tex_color[2]);
}

//reads a given line for an double and cuts out any intro text  (with greater than conditions)
double readFullLineDouble(std::string line, int skip_factor, std::string type, double condition, std::string conditions) {
	line.erase(0, skip_factor);
	double val;
	if (!(val = stod(line)) || val < condition) {
		std::cerr << "ERROR::Unable to read " << type << " from file, please input a double " << conditions << std::endl;
	}
	return val;
}

//reads a given line for an double and cuts out any intro text (without conditions)
double readFullLineDouble(std::string line, int skip_factor, std::string type) {
	line.erase(0, skip_factor);
	double val;
	if (!(val = stod(line))) {
		std::cerr << "ERROR::Unable to read " << type << " from file, please input a double" << std::endl;
	}
	return val;
}

//reads a given line for an integer and cuts out any intro text 
int readFullLineInt(std::string line, int skip_factor, std::string type, int condition, std::string conditions) {
	line.erase(0, skip_factor);
	double val;
	if (!(val = stoi(line)) || val < condition) {
		std::cerr << "ERROR::Unable to read " << type << " from file, please input a double " << conditions << std::endl;
	}
	return val;
}

//loads input files with path to file and pointers to structs to save info to, returns the timestep
double loadFile(std::string input_file_path, double y[], integratorCreateInfo* create_info, renderCreateInfo* render_info) {
	std::ifstream input_file;

	std::cout << "Loading data from file: " << input_file_path << " ..." << std::endl;

	//open file
	input_file.open(input_file_path);

	//check file exists and throw error if not
	if (!input_file) {
		std::cerr << "ERROR: Unable to open the input file at: " << input_file_path << std::endl;
		exit(1);
	}

	//read each line of the file
	std::vector<std::string> lines;
	std::string line;
	while (getline(input_file, line)) {
		lines.push_back(line);
	}

	//find end time
	create_info->end_time = readFullLineDouble(lines[0], 9, "end time", 0, "above (or equal) zero");

	//find timestep
	double timestep = readFullLineDouble(lines[1], 17, "timestep", 0, "above (or equal) zero");

	//find max timestep
	create_info->max_timestep = readFullLineDouble(lines[2], 13, "maximum timestep", 0, "above (or equal) zero");

	//find error tolerance
	create_info->err_tolerance = readFullLineDouble(lines[3], 16, "error tolerance", 0, "above (or equal) zero");

	//find Black Hole Radius
	create_info->R_s = readFullLineDouble(lines[4], 21, "Schwarzschild radius", 0, "above (or equal) zero");

	//find simulation type
	lines[7].erase(0, 14);
	std::cout << lines[7] << std::endl;
	if (lines[7] == "Circle 2D") {
		//check whether to output to console
		int output = readFullLineInt(lines[10], 15, "console output", 0, "either 0 or 1");
		create_info->console_output = output == 1 ? true : false;

		//output file
		create_info->saving_to_file = true;
		lines[11].erase(0, 13);
		std::stringstream output_data(lines[11]);

		if (!(output_data >> create_info->filepath)) {
			std::cerr << "ERROR::Unable to read saving filepath" << std::endl;
		}

		//find Radial Pos
		double rad_coord = readFullLineDouble(lines[12], 13, "radial position", create_info->R_s, "above (or equal) the schwarzschild radius");

		y[0] = 0.0;
		y[1] = 0.0;
		y[2] = rad_coord;
		y[3] = 0.0;
		y[4] = 0.0;
		y[5] = findCircularAngularSpeeed(create_info->R_s, rad_coord);
		y[1] = findInitialTimeDifferential2D(create_info->R_s, y, false);

		//output initials to user
		std::cout << "======================================================================" << std::endl;
		std::cout << "R_s: " << create_info->R_s << "\t\t Max timestep: " << create_info->max_timestep
			<< "\t\t End time: " << create_info->end_time << "\t\t Error tolerance: " << create_info->err_tolerance
			<< "\t\t Initial conditions(t_dot,r,r_dot,phi,phi_dot): " << y[1] << "\t" << y[2] << "\t"
			<< y[3] << "\t" << y[4] << "\t" << y[5] << "\t\t Console output: " << create_info->console_output
			<< "\t\t Output data path:" << create_info->filepath << std::endl;
		std::cout << "======================================================================" << std::endl;
	}
	else if (lines[7] == "Input 2D") {
		//check whether to output to console
		int output = readFullLineInt(lines[10], 15, "console output", 0, "either 0 or 1");
		create_info->console_output = output == 1 ? true : false;

		//output file
		create_info->saving_to_file = true;
		lines[11].erase(0, 13);
		std::stringstream output_data(lines[11]);

		if (!(output_data >> create_info->filepath)) {
			std::cerr << "ERROR::Unable to read saving filepath" << std::endl;
		}


		y[0] = 0.0;
		y[1] = 0.0;
		//find initial system conditions and create array
		y[2] = readFullLineDouble(lines[12], 13, "radial position", create_info->R_s, "above (or equal) the schwarzschild radius");
		y[3] = readFullLineDouble(lines[13], 16, "radial velocity");
		y[4] = readFullLineDouble(lines[14], 10, "phi position");
		y[5] = readFullLineDouble(lines[15], 13, "phi velocity");
		y[1] = findInitialTimeDifferential2D(create_info->R_s, y, false);

		//output initials to user
		std::cout << "======================================================================" << std::endl;
		std::cout << "R_s: " << create_info->R_s << "\t\t Max timestep: " << create_info->max_timestep
			<< "\t\t End time: " << create_info->end_time << "\t\t Error tolerance: " << create_info->err_tolerance
			<< "\t\t Initial conditions(t_dot,r,r_dot,phi,phi_dot): " << y[1] << "\t" << y[2] << "\t"
			<< y[3] << "\t" << y[4] << "\t" << y[5] << "\t\t Console output: " << create_info->console_output
			<< "\t\t Output data path:" << create_info->filepath << std::endl;
		std::cout << "======================================================================" << std::endl;
	}
	else if (lines[7] == "Render 3D") {
		//get cam pos
		lines[10].erase(0, 11);
		std::stringstream cam_data(lines[10]);
		
		if (!(cam_data >> render_info->cam_pos.x)) {
			std::cerr << "ERROR::Unable to read radial postion" << std::endl;
		}

		if (!(cam_data >> render_info->cam_pos.y)) {
			std::cerr << "ERROR::Unable to read theta postion" << std::endl;
		}

		if (!(cam_data >> render_info->cam_pos.z)) {
			std::cerr << "ERROR::Unable to read phi postion" << std::endl;
		}

		//get screen size
		lines[11].erase(0, 12);
		std::stringstream screen_data(lines[11]);

		if (!(screen_data >> render_info->screen_size.x)) {
			std::cerr << "ERROR::Unable to read radial postion" << std::endl;
		}

		if (!(screen_data >> render_info->screen_size.y)) {
			std::cerr << "ERROR::Unable to read theta postion" << std::endl;
		}

		//get screen scale
		render_info->screen_scale = readFullLineInt(lines[12], 13, "screen scale", 1, "larger than 0");

		//get sphere texture
		lines[13].erase(0, 20);
		render_info->sphere_texture = cv::imread(lines[13].c_str(), cv::IMREAD_COLOR_BGR);

		//check its loaded
		if (render_info->sphere_texture.empty()) {
			std::cerr << "ERROR::Failed to load texture image" << std::endl;
		}

		create_info->is3D = true;

		//output initials to user
		std::cout << "======================================================================" << std::endl;
		std::cout << "R_s: " << create_info->R_s << "\t\t Max timestep: " << create_info->max_timestep
			<< "\t\t End time: " << create_info->end_time << "\t\t Error tolerance: " << create_info->err_tolerance
			<< "\t\t Camera position: " << render_info->cam_pos.x << "\t" << render_info->cam_pos.y << "\t"
			<< render_info->cam_pos.z << "\t\t Screen size, scale: " << render_info->screen_size.x << "\t"
			<< render_info->screen_size.y << "\t" << render_info->screen_scale << "\t\t Sphere texture path: "
			<< lines[13] << std::endl;
		std::cout << "======================================================================" << std::endl;

	}
	else {
		std::cerr << "ERROR::Unable to read simulation type please use one of the following:" << std::endl;
		std::cout << "\t\tInput 2D" << std::endl;
		std::cout << "\t\tCircle 2D" << std::endl;
		std::cout << "\t\tRender 3D" << std::endl;
	}

	//any user input will contiue application if they are happy
	std::cout << "Press any key to continue..." << std::endl;
	char val;
	std::cin >> val;

	std::cout << "Successfully loaded file" << std::endl;
	return timestep;
}

int main() {
	//get input file to load from user
	std::string input_file_path;

	std::cout << "Where is the input file: " << std::endl;
	std::cin >> input_file_path;

	//load input file and get initials
	double y[6];
	integratorCreateInfo integrator_info;
	renderCreateInfo render_info;
	double timestep = loadFile(input_file_path, y, &integrator_info, &render_info);

	//run simulation
	if (!integrator_info.is3D) {
		if (individualIntegration(y, timestep, &integrator_info) == 1) {
			std::cout << "ERROR::Failed to integrate" << std::endl;
		}
	}
	else {
		//image loading
		cv::Mat image(render_info.screen_size.x * render_info.screen_scale, render_info.screen_size.y * render_info.screen_scale, CV_8UC3, cv::Scalar(50, 200, 150));

		//display it to screen "WINDOW"
		cv::namedWindow("WINDOW", cv::WINDOW_AUTOSIZE);
		cv::imshow("WINDOW", image);
		cv::waitKey(1);

		//iterate for each pixel in parallel
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < render_info.screen_size.x; i++) {
			for (int j = 0; j < render_info.screen_size.y; j++) {

				if (integrator_info.console_output) std::cout << "Starting Ray: " << i << j << std::endl;

				//setup ray
				double z[8];
				findRayDirection(z, render_info.cam_pos, j, i, 114, render_info.screen_size.y, render_info.screen_size.x, integrator_info.R_s);

				//integrate ray
				int success = individualIntegration(z, timestep, &integrator_info);
				if(success == 1) {
					std::cout << "ERROR::Failed to integrate pixel: " << i << ", " << j << std::endl;
				}

				//find ray color
				cv::Scalar color;
				if (success == 2) {
					color = cv::Scalar(0, 0, 0);
				}
				else if (z[2] > integrator_info.R_s) {
					color = sampleTexture(render_info.sphere_texture, z);
				}
				else {
					color = cv::Scalar(0, 0, 0);
				}

				//draw rectangle
				cv::rectangle(image, cv::Point(j * render_info.screen_scale, i * render_info.screen_scale), 
					cv::Point((j + 1) * render_info.screen_scale, (i + 1) * render_info.screen_scale), color, cv::FILLED, cv::LINE_8);
				if (j % render_info.screen_size.x == 0) {
					cv::imshow("WINDOW", image);
					cv::waitKey(1);
				}
			}
		}
		cv::imwrite("output_image.png", image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	return 0;
}