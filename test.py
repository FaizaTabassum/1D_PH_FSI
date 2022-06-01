xs = np.linspace(0, self.tube_length.val, int(self.number_sections.val))

        point_start_linear_expansion = self.point_start_expansion.val + self.transition_window.val
        point_end_linear_expansion = self.point_end_expansion.val - self.transition_window.val


        length_expansion = self.point_end_expansion.val - self.point_start_expansion.val
        radius_end = np.tan(np.radians(self.expansion_angle.val)) * length_expansion + self.tube_base_radius.val

        m = (radius_end - self.tube_base_radius.val) / length_expansion
        ycross = self.tube_base_radius.val - m * self.point_start_expansion.val
        value_point_start_linear_expansion = m * (point_start_linear_expansion) + ycross
        value_point_end_linear_expansion = m * (point_end_linear_expansion) + ycross

        a, b, c, d,e = sym.symbols(['a', 'b', 'c', 'd', 'e'])
        y1_1 = a *  self.point_start_expansion.val ** 4 + b *  self.point_start_expansion.val ** 3 + c *  self.point_start_expansion.val**2 + d*self.point_start_expansion.val +e
        y1_2 = 4 * a *  self.point_start_expansion.val ** 3 + 3 * b *  self.point_start_expansion.val**2 + 2*c*self.point_start_expansion.val + d
        y1_3 = a * point_start_linear_expansion ** 4 + b * point_start_linear_expansion ** 3 + c * point_start_linear_expansion**2 + d*point_start_linear_expansion +e
        y1_4 = 4 * a * point_start_linear_expansion ** 3 + 3 * b * point_start_linear_expansion**2 + 2*c*point_start_linear_expansion +d
        y1_5 = 12 * a * self.point_start_expansion.val ** 2 + 6 * b * self.point_start_expansion.val + 2 * c
        sol_start = sym.solve(
            [y1_1 - self.tube_base_radius.val, y1_2 - 0, y1_3 - value_point_start_linear_expansion, y1_4 - m, y1_5 -0],
            dict=True)
        sol_start = sol_start[0]
        a, b, c, d,e = sym.symbols(['a', 'b', 'c', 'd', 'e'])
        y2_1 = a * self.point_end_expansion.val ** 4 + b * self.point_end_expansion.val ** 3 + c * self.point_end_expansion.val**2 + d*self.point_end_expansion.val +e
        y2_2 = 4 * a * self.point_end_expansion.val ** 3 + 3 * b * self.point_end_expansion.val**2+ 2*c*self.point_end_expansion.val + d
        y2_3 = a * point_end_linear_expansion ** 4 + b * point_end_linear_expansion ** 3 + c * point_end_linear_expansion**2 + d*point_end_linear_expansion +e
        y2_4 = 4 * a * point_end_linear_expansion ** 3 + 3 * b * point_end_linear_expansion**2 + c*point_end_linear_expansion +d
        y2_5 = 12 * a * self.point_end_expansion.val ** 2 + 6 * b * self.point_end_expansion.val + 2 * c


        sol_end = sym.solve([y2_1 - radius_end, y2_2 - 0, y2_3 - value_point_end_linear_expansion, y2_4 - m, y2_5-0],
                        dict=True)

        sol_end = sol_end[0]

        xstraight_entrance = np.linspace(0, self.point_start_expansion.val, 50)
        ystraight_entrance = np.ones(50) * self.tube_base_radius.val

        xtrans_entrance = np.linspace(self.point_start_expansion.val, point_start_linear_expansion, 50)
        ytrans_entrance = sol_start[a] * xtrans_entrance ** 4 + sol_start[b] * xtrans_entrance ** 3 + sol_start[c] * xtrans_entrance**2 + sol_start[d]*xtrans_entrance + sol_start[e]

        xexp = np.linspace(point_start_linear_expansion,point_end_linear_expansion, 50)
        yexp = m * xexp + ycross

        xtrans_exit = np.linspace(point_end_linear_expansion, self.point_end_expansion.val, 50)
        ytrans_exit = sol_end[a] * xtrans_exit ** 4 + sol_end[b] * xtrans_exit ** 3 + sol_end[c] * xtrans_exit**2 + sol_end[d]*xtrans_exit + sol_end[e]

        xstraight_exit = np.linspace(self.point_end_expansion.val, self.tube_length.val, 50)
        ystraight_exit = np.ones(50) * radius_end