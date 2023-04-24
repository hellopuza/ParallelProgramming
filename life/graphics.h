#include <SFML/Graphics.hpp>

void display(sf::RenderWindow* window, int8_t* grid_data, int grid_size)
{
    sf::RectangleShape rect;
    rect.setSize(sf::Vector2f(static_cast<float>(window->getSize().x) / static_cast<float>(grid_size),
                              static_cast<float>(window->getSize().y) / static_cast<float>(grid_size)));

    for (int y = 0; y < grid_size; y++)
    {
        for (int x = 0; x < grid_size; x++)
        {
            rect.setPosition(sf::Vector2f(rect.getSize().x * x, rect.getSize().y * y));
            rect.setFillColor(grid_data[grid_size * y + x] ? sf::Color::White : sf::Color::Black);
            window->draw(rect);
        }
    }
    window->display();
}