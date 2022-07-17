import json


def update_json_file(ficheiro, left_size, bottom_size, right_size, top_size, ll_angle, ll_m, ll_b, ll_mse_error, bl_angle, bl_m, bl_b, bl_mse_error, rl_angle, rl_m, rl_b, rl_mse_error, tl_angle, tl_m, tl_b, tl_mse_error, DISTANCE_BETWEEN_POINTS, bot_left, bot_right, top_right, top_left, trench_depth, W_Number, W_Row, W_Trench_Number):

    entry = {
                'file_name': ficheiro,
                'Wafer_Number': W_Number,
                'Wafer_Row': W_Row,
                'Number_of_Trench': W_Trench_Number,
                'points_dist': DISTANCE_BETWEEN_POINTS,
                'left_size': float(left_size*DISTANCE_BETWEEN_POINTS),
                'bottom_size': float(bottom_size*DISTANCE_BETWEEN_POINTS),
                'right_size': float(right_size*DISTANCE_BETWEEN_POINTS),
                'top_size': float(top_size*DISTANCE_BETWEEN_POINTS),
                'll_angle': ll_angle,
                'll_m': ll_m,
                'll_b': ll_b,
                'll_mse_error': ll_mse_error,
                'bl_angle': bl_angle,
                'bl_m': bl_m,
                'bl_b': bl_b,
                'bl_mse_error': bl_mse_error,
                'rl_angle': rl_angle,
                'rl_m': rl_m,
                'rl_b': rl_b,
                'rl_mse_error': rl_mse_error,
                'tl_angle': tl_angle,
                'tl_m': tl_m,
                'tl_b': tl_b,
                'tl_mse_error': tl_mse_error,
                'bottom_left': bot_left,
                'bottom_right': bot_right,
                'top_left': top_left,
                'top_right': top_right,
                'trench_depth': float(trench_depth*DISTANCE_BETWEEN_POINTS)
            }

    return entry


def save_json_file(data, WAFER_NUMBER):

    filename = "DRIE_Parametric_Study/Wafer"+str(WAFER_NUMBER)+"_/Data_WAFER"+str(WAFER_NUMBER)+".json"

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
"""
    with open(filename, 'r') as file:
        data = json.load(file)

        print(type(data))

    data.append(entry)

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

"""
