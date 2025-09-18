"""
Tests basiques pour vérifier que la configuration de test fonctionne
"""

import pytest


def test_basic_math():
    """Test basique pour vérifier que pytest fonctionne"""
    assert 1 + 1 == 2
    assert 2 * 3 == 6


def test_string_operations():
    """Test des opérations sur les chaînes"""
    text = "Magic Pong"
    assert text.lower() == "magic pong"
    assert len(text) == 10
    assert "Magic" in text


def test_list_operations():
    """Test des opérations sur les listes"""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5


class TestBasicClass:
    """Classe de test pour vérifier la structure"""

    def test_class_method(self):
        """Test d'une méthode de classe"""
        assert True

    def test_with_setup(self):
        """Test avec setup"""
        data = {"key": "value"}
        assert data["key"] == "value"
        assert "key" in data


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
        (4, 16),
    ],
)
def test_square_function(input_value, expected):
    """Test paramétrisé pour une fonction carré"""

    def square(x):
        return x * x

    assert square(input_value) == expected


def test_exception_handling():
    """Test de gestion des exceptions"""
    with pytest.raises(ZeroDivisionError):
        1 / 0

    with pytest.raises(ValueError):
        int("not_a_number")
