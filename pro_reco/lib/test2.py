class OuterClass:
    class NestedClass:
        def __init__(self, value):
            self.value = value

        def display(self):
            print(f"Value is {self.value}")

    def __init__(self, value):
        self.outer_value = value
        self.nested_instance = self.NestedClass(value * 2)

    def display(self):
        print(f"Outer value is {self.outer_value}")
        self.nested_instance.display()

# 사용 예시
outer_instance = OuterClass(10)
outer_instance.display()