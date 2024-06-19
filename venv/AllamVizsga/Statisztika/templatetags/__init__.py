from django import template

register = template.Library()

@register.filter(name='enumerate')
def enumerate_filter(value):
    return list(enumerate(value))
