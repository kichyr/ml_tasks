import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple
import copy


def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k

class DecisionTree:
    """
    Класс решающего дерева
    """

    def __init__(
            self,
            max_depth: int,
            min_node_size: int
    ) -> None:
        """
        Конструктор класса решающего дерева
        :param max_depth: максимальная глубина дерева
        :param min_node_size: минимальный размер узла (количества объектов в нем)
        """
        self.cat_feature_value_list: List = []
        self.max_depth: int = max_depth
        self.min_node_size: int = min_node_size
        self.tree = None
        self.column_of_features: pd.core.indexes.base.Index = None
        self.categorical_features_list: List = []
        self.categorical_feature_index_list: List = []
        self.dict_index_to_num_cat_feature = {}
        self.root: DecisionTree.TreeBinaryNode = DecisionTree.TreeBinaryNode()

    class TreeBinaryNode:
        """
        Класс узла решающего дерева
        """

        def __init__(self, left_node=None, right_node=None, data_dict=None, list_of_sorted_cat_features=None) -> None:
            """
            Конструктор класса узла
            :param left_node: левый ребенок узла, которым может быть как следующим узлом, так и просто числом,
            определяющим класс данных, если построение дерева окончено
            :param right_node: правый ребенок узла, которым может быть как следующим узлом, так и просто числом,
            определяющим класс данных
            :param data_dict: словарь с данными узла, содержащий индекс фичи, пороговое значение и разбиваемые группы
            """
            if data_dict is None:
                data_dict = {}
            if list_of_sorted_cat_features is None:
                list_of_sorted_cat_features = []
            self.left: Union[DecisionTree.TreeBinaryNode, int] = left_node
            self.right: Union[DecisionTree.TreeBinaryNode, int] = right_node
            self.data_dict: Dict = data_dict
            self.sorted_cat_features: List[List[Dict]] = list_of_sorted_cat_features

        def set_left_as_node(self, left_node) -> None:
            """
            Метод, записывающий в левого ребенка типа узел подаваемый на вход узел
            """
            self.left: DecisionTree.TreeBinaryNode = left_node

        def set_left_as_num(self, left_num: int):
            """
            Метод, записывающий в левого ребенка типа int подаваемое на вход число
            """
            self.left: int = left_num

        def set_right_as_node(self, right_node):
            """
            Метод, записывающий в правого ребенка типа узел подаваемый на вход узел
            """
            self.right: DecisionTree.TreeBinaryNode = right_node

        def set_right_as_num(self, right_num: int):
            """
            Метод, записывающий в правого ребенка типа int подаваемое на вход число
            """
            self.right: int = right_num

    class TreeNonBinaryNode:
        def __init__(self):
            pass

    @staticmethod
    def calc_gini(
            groups: List[List],
            classes: set
    ) -> float:
        """
        Метод для вычисления критерия информативности Джини
        :param groups: входные группы с данными, среди которых могут быть представители всех классов.
        Последняя ячейка хранит его номер
        :param classes: существующие классы
        :return: значение критерия информативности
        """
        # Количество элементов в узле (во всех группах)
        num_of_elements: int = sum([len(group) for group in groups])
        gini: float = 0.0

        for group in groups:
            size = len(group)

            # Если группа пустая, скипаем ее
            if size == 0:
                continue

            score: float = 0.0

            # Пробегаемся по классам и сравниваем классы содержмого с ними
            for class_i in classes:
                p: float = 0.
                for index, element in enumerate(group):
                    # print(index)
                    # Если классы совпали, инкрементируем вероятность ...
                    if element[-1] == class_i:
                        p += 1

                # ... и нормируем ее
                p = p / np.double(size)
                score += p * (1 - p)

            # Суммируем критерий информативности
            gini += score * size / float(num_of_elements)

        return gini

    @staticmethod
    def do_single_split(
            index: int,
            threshold: float,
            data: List
    ) -> List[List]:
        """
        Метод для построения одного разбиения на основе уже вычисленного порога некоторой фичи
        :param index: индекс фичи
        :param threshold: пороговое значение
        :param data: список данных (иначе говоря - некоторый массив строк row исходной таблицы с данными)
        :return: два массива-ребенка
        """
        # Создаем данные для левого и правого ребенка
        left: List = []
        right: List = []

        # Пробегаемся по всем элеменным таблицы и разбиваем на детей в зависимости от значения в ячейке и порога
        for row in data:
            if row[index] < threshold:
                left.append(row)
            else:
                right.append(row)

        return [left, right]

    def do_full_one_node_split(
            self,
            data: List
    ) -> Tuple[Dict[str, Any], List[List[Dict]]]:
        """
        Произвести полное разбиение для входного массива строк row таблицы: пробежаться по всем фичам и соответствующим
        ячейкам с вычислением Джини для каждого случая. Выбирается наилучший Джини
        :param data: массив строк row с данными. Последняя ячейка хранит номер класса
        :return: словарь с индексом фичи, порогом и разбиением
        """

        # Создаем множество со всеми имеющимися классами в data
        classes: set = set(row[-1] for row in data)

        # Инициализируем используемые данные
        split_index, split_threshold, best_gini, best_split = sys.maxsize, sys.maxsize, sys.maxsize, None
        out: Dict = {}

        list_of_dict_cat_to_int, list_of_dict_int_to_cat = self.create_dict_for_cat_features(data)

        # Пробегаемся по всем фичам и ячейкам для вычисления наилучшего Джини
        for row in data:
            for index in range(len(data[0]) - 1):
                if index in self.categorical_feature_index_list:
                    # Строим конкретное разбиение для текущего значения index и data
                    feature_as_str = row[index]

                    cat_num = self.dict_index_to_num_cat_feature[index]
                    cat_dict = list_of_dict_cat_to_int[cat_num]
                    new_data = self.replace_categorical_str_to_int(data, cat_dict, index)
                    c = cat_dict[feature_as_str]
                    groups = self.do_single_split(index, c, new_data)

                    # Вычисляем для него Джини
                    gini = self.calc_gini(groups, classes)
                else:
                    # Строим конкретное разбиение для текущего значения index и data
                    groups = self.do_single_split(index, row[index], data)

                    # Вычисляем для него Джини
                    gini = self.calc_gini(groups, classes)

                # Если Джини лучше предыдущих, сохраняем его и соответствующие ему индекс, порог и разбиение
                if gini < best_gini:
                    split_index, split_threshold, best_gini, best_split = index, row[index], gini, groups

        if split_index in self.categorical_feature_index_list:
            # for feature_index in self.categorical_feature_index_list:
            cat_num = self.dict_index_to_num_cat_feature[split_index]
            cat_dict = list_of_dict_int_to_cat[cat_num]
            best_split[0] = self.replace_int_to_categorical_str(best_split[0], cat_dict, split_index)
            best_split[1] = self.replace_int_to_categorical_str(best_split[1], cat_dict, split_index)

        # Заполнение данных для вывода
        out['index'] = split_index
        out['threshold'] = split_threshold
        out['groups'] = best_split

        return out, [list_of_dict_cat_to_int, list_of_dict_int_to_cat]

    @staticmethod
    def create_value_of_last_node(group: List) -> int:
        """
        Метод, создающий финальное значение в листе на основе наиболее часто встречающегося класса в группе
        :param group: список строк row исходной таблицы
        :return: наиболее часто встречающийся класс
        """
        classes = [row[-1] for row in group]
        return max(set(classes), key=classes.count)

    def do_split(self, node: TreeBinaryNode, current_depth: int) -> None:
        """
        Рекурсивный метод для построения дерева
        :param node: входной узел
        :param current_depth: текущая глубина узла node
        :return: None
        """
        # Из текущего узла вытаскиваем уже найденные данные с группами
        left_list, right_list = node.data_dict['groups']

        # Удаляем содержимое словаря по ключу групп, так как нам эти данные больше не нужны
        del (node.data_dict['groups'])

        # Остановка построения дерева, если один из детей пустой
        if not left_list or not right_list:
            node.left = node.right = self.create_value_of_last_node(left_list + right_list)
            return

        # Остановка построения дерева, если глубина достаточно большая
        if current_depth >= self.max_depth:
            node.set_left_as_num(self.create_value_of_last_node(left_list))
            node.set_right_as_num(self.create_value_of_last_node(right_list))
            return

        # Вызов рекурсивной части, если не было выполнено одно из условий остановки
        node.left = self.do_recurse(data_list=left_list, depth=current_depth)
        node.right = self.do_recurse(data_list=right_list, depth=current_depth)

    def do_recurse(self, data_list: List, depth: int):
        """
        Рекурсивная часть метода do_split. Остановка вычислений, если текущий размер узла меньше минимального, либо
        вызов do_split
        :param data_list: список данных группы
        :param depth: текущая глубина
        :return: либо номер класса, если рекурсию необходимо завершить, либо узел TreeBinaryNode
        """

        # Инициализация узла
        node: Union[DecisionTree.TreeBinaryNode, int]

        # Если текущий размер узла меньше минимального, записываем номер класса в узел. Иначе - вызов do_split
        if len(data_list) <= self.min_node_size:
            node: int = self.create_value_of_last_node(data_list)
        else:
            data_dict, list_of_dicts_to_convert = self.do_full_one_node_split(data_list)
            node: DecisionTree.TreeBinaryNode = self.TreeBinaryNode(left_node=None, right_node=None,
                                                                    data_dict=data_dict,
                                                                    list_of_sorted_cat_features=list_of_dicts_to_convert
                                                                    )
            self.do_split(node=node, current_depth=depth + 1)

        return node

    def build_tree(self, data: List):
        """
        Метод для построения дерева на основе входной таблицы, содержащей выборку с X и Y. Необходимо предварительная
        обработка с помещением номера класса объекта в конец таблицы
        :param data: список входных данных. Последний столбец - номер класса
        :return: узел корня с содержащимся в нем деревом
        """
        root = self.do_full_one_node_split(data)
        root_node: DecisionTree.TreeBinaryNode = self.TreeBinaryNode(left_node=None, right_node=None, data_dict=root)
        self.do_split(root_node, 1)
        return root_node

    def fit(self,
            x: pd.core.frame.DataFrame,
            y: pd.core.series.Series):
        """
        Построение дерева на основе разбитой выборки DataFrame
        :param x: выборка x для построения дерева
        :param y: выборка y для построения дерева
        :return: корневой узел дерева
        """
        assert type(x) == pd.core.frame.DataFrame, "Некорректный тип данных x_train"
        assert type(y) == pd.core.series.Series, "Некорректный тип данных y_train"

        self.set_categorical_data_from_dataframe(x)

        # Обернем все в numpy
        x_numpy = x.to_numpy()
        y_numpy = y.to_numpy()
        y_numpy = y_numpy.reshape((1, len(y_numpy))).transpose().astype(int)
        dataset = np.concatenate((x_numpy, y_numpy), axis=1)

        # Вызовем простроение дерева
        root_1, root_2 = self.do_full_one_node_split(list(dataset))
        root_node = self.TreeBinaryNode(left_node=None, right_node=None, data_dict=root_1,
                                        list_of_sorted_cat_features=root_2)
        self.do_split(root_node, 1)
        self.root = root_node
        return root_node

    def single_predict(self, node, row: List) -> object:
        """
        Метод для предсказания класса некоторого элемента
        :rtype: DecisionTree.TreeBinaryNode
        :param node: узел
        :param row: строка таблицы для предсказания класса
        :return: предсказанное значение класса
        """

        index = node.data_dict['index']

        # Если признак элемента является категориальным, то находим соответствующее ему число
        if index in self.categorical_feature_index_list:
            cat_num = self.dict_index_to_num_cat_feature[index]
            cat_dict_str_to_int = node.sorted_cat_features[0][cat_num]
            threshold = cat_dict_str_to_int[node.data_dict['threshold']]

            if row[index] in cat_dict_str_to_int:
                current_value = cat_dict_str_to_int[row[index]]
            else:
                current_value = 0
        else:
            current_value = row[index]
            threshold = node.data_dict['threshold']

        # Определяем, куда следует отнести элемент в соответствии с отношением к порогу - влево или направо
        if current_value < threshold:
            if type(node.left) == DecisionTree.TreeBinaryNode:
                return self.single_predict(node=node.left, row=row)
            else:
                return node.left
        else:
            if type(node.right) == DecisionTree.TreeBinaryNode:
                return self.single_predict(node=node.right, row=row)
            else:
                return node.right

    def predict(self, x_test: pd.core.frame.DataFrame) -> np.ndarray:
        """
        Метод для предсказания классов элементов входной таблицы DataFrame
        :param x_test: тестовая таблица с элементами
        :return: массив с предсказанными классами
        """

        # Перегоняем все в numpy, готовим массивы
        x_numpy = x_test.to_numpy()
        dataset = x_numpy
        predictions = np.zeros(len(dataset), dtype=np.int64)

        # Пробегаемся по всем строками входной таблицы и сравниваем классы
        for row_index, row in enumerate(dataset):
            predictions[row_index] = self.single_predict(self.root, row)

        return predictions

    def draw(self, node, columns: pd.core.indexes.base.Index, current_depth):
        """
        Метод для отрисовки дерева
        :param node: текущий узел. В первый раз на вход подается корень
        :param columns: список фич из dataFrame
        :param current_depth: текущая глубина
        """

        if type(node) == DecisionTree.TreeBinaryNode:
            if node.data_dict['index'] in self.categorical_feature_index_list:
                index = node.data_dict['index']
                cat_num = self.dict_index_to_num_cat_feature[index]
                cat_dict_int_to_str = node.sorted_cat_features[1][cat_num]
                cat_dict_str_to_int = node.sorted_cat_features[0][cat_num]
                current_int = cat_dict_str_to_int[node.data_dict['threshold']]
                variants_int = np.arange(current_int)
                variants_str = []
                for var in variants_int:
                    variants_str.append(cat_dict_int_to_str[var])

                print("—" * current_depth, columns[node.data_dict['index']], " = ", ' || '.join(variants_str))
                self.draw(node.left, columns, current_depth + 1)
                self.draw(node.right, columns, current_depth + 1)
            else:
                print("—" * current_depth, columns[node.data_dict['index']], "<", node.data_dict['threshold'])
                self.draw(node.left, columns, current_depth + 1)
                self.draw(node.right, columns, current_depth + 1)
        else:
            print("—" * current_depth, node)

    def set_categorical_data_from_dataframe(self, data: pd.core.frame.DataFrame) -> None:
        """
        Препроцессорный метод для инициализации словарей, используемых для работы с категориальными фичами
        :param data: входная таблица с данными
        """

        # В методе заполняются листы и словари (поля класса):
        # лист со всеми категориальными фичами текущей таблицы данных...
        self.categorical_features_list = data.columns[data.dtypes == object].tolist()

        a = np.arange(len(data.columns))
        categorical_feature_index = []

        # ... словарь для перехода от номера категориальной фичи среди колонок таблицы данных
        # к номеру в листе с категориальными фичами
        for i, feature in enumerate(self.categorical_features_list):
            categorical_feature_index.append(a[feature == data.columns][0])
            self.dict_index_to_num_cat_feature[categorical_feature_index[-1]] = i

        # ... словарь с номерами категориальных фич среди всех фич таблицы
        self.categorical_feature_index_list = categorical_feature_index

        assert len(self.cat_feature_value_list) == 0

        # ... лист со всеми принимаемыми значениями всех фич
        for feature in self.categorical_features_list:
            self.cat_feature_value_list.append(set(data[feature]))

    def create_dict_for_cat_features(self, x_m: List) -> Tuple[List[Dict[Any, int]], List[Dict[int, Any]]]:
        """
        Особый метод для составления порядка значений некоторой категориальной фичи. Значения категориального признака
        сортируются по возрастанию доли объектов +1 класса среди объектов выборки x_m с соответствующим значением
        этого признака
        :param x_m: некоторая выборка
        :return: два листа, характеризующие сортировку категориальных признаков для текущего узла
        """

        list_of_dict_of_norms = []
        sorted_norms = []
        list_of_dict_cat_to_int = []
        list_of_dict_int_to_cat = []

        # Бежим по всем фичам и принимаемым значениям
        for id_feature, cat_feature_dict in enumerate(self.cat_feature_value_list):

            dict_of_c_sizes = {key: 0 for key in cat_feature_dict}
            dict_of_c_first_class_sizes = {key: 0 for key in cat_feature_dict}
            dict_of_norms = {}
            dict_cat_to_int = {}
            dict_int_to_cat = {}

            # Сперва строим словарь с нормами - долей объектов первого класса среди объектов выборки с соответствующим
            # значением этого признака
            for row in x_m:
                row_c = row[self.categorical_feature_index_list[id_feature]]
                dict_of_c_sizes[row_c] += 1

                if row[-1] == 1:
                    dict_of_c_first_class_sizes[row_c] += 1

            for key in dict_of_c_sizes.copy():
                if dict_of_c_sizes[key] == 0:
                    del dict_of_c_sizes[key]
                    del dict_of_c_first_class_sizes[key]
                    continue

                dict_of_norms[key] = dict_of_c_first_class_sizes[key] / dict_of_c_sizes[key]

            # Помещаем словарь в список (одна фича - один словарь)
            list_of_dict_of_norms.append(dict_of_norms)

            # Сортируем нормы и формируем на их основе словари
            a = []
            for key in dict_of_norms:
                a.append(dict_of_norms[key])
            a.sort()
            sorted_norms.append(a)

            i = 0
            for norm in a:
                key = get_key(dict_of_norms, norm)
                del dict_of_norms[key]

                dict_cat_to_int[key] = i
                dict_int_to_cat[i] = key
                i += 1

            list_of_dict_cat_to_int.append(dict_cat_to_int)
            list_of_dict_int_to_cat.append(dict_int_to_cat)

        return list_of_dict_cat_to_int, list_of_dict_int_to_cat

    @staticmethod
    def replace_categorical_str_to_int(data: List, dict_cat_to_int: Dict[Any, int], index: int) -> List:
        """
        Метод для замены значений категориального признака в лице строки на инт
        :param data: таблица с данными
        :param dict_cat_to_int: словарь для перевода значений категориального признака из строк в инты
        :param index: индекс фичи
        :return: новая таблица, в категориальной фиче под индексом index которой находятся инты
        """
        new_data = copy.deepcopy(data)
        for row_id, row in enumerate(new_data):
            cat = row[index]
            new_data[row_id][index] = dict_cat_to_int[cat]

        return new_data

    @staticmethod
    def replace_int_to_categorical_str(data: List, dict_int_to_cat: Dict[int, Any], index: int) -> List:
        """
        Метод для замены значений категориального признака в лице строки на инт
        :param data: таблица с данными
        :param dict_int_to_cat: словарь для перевода значений категориального признака из интов в строки
        :param index: индекс фичи
        :return: новая таблица, в категориальной фиче под индексом index которой находятся строки
        """
        new_data = copy.deepcopy(data)
        for row_id, row in enumerate(new_data):
            i = row[index]
            new_data[row_id][index] = dict_int_to_cat[i]

        return new_data