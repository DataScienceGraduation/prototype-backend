from django.contrib import admin
from .models import Report, ChartData, DataInsight, ReportTemplate

@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'model_entry', 'report_type', 'status', 'created_at']
    list_filter = ['report_type', 'status', 'created_at']
    search_fields = ['title', 'model_entry__name']
    readonly_fields = ['created_at', 'updated_at']

    fieldsets = (
        ('Basic Information', {
            'fields': ('model_entry', 'title', 'description', 'report_type')
        }),
        ('Status', {
            'fields': ('status',)
        }),
        ('AI Insights', {
            'fields': ('ai_insights',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(ChartData)
class ChartDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'chart_type', 'report', 'created_at']
    list_filter = ['chart_type', 'created_at']
    search_fields = ['title', 'report__title']
    readonly_fields = ['created_at']

    fieldsets = (
        ('Basic Information', {
            'fields': ('report', 'chart_type', 'title', 'description')
        }),
        ('Chart Data', {
            'fields': ('chart_data', 'chart_config'),
            'classes': ('collapse',)
        }),
        ('Generated Content', {
            'fields': ('chart_image_base64', 'chart_html'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

@admin.register(DataInsight)
class DataInsightAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'insight_type', 'report', 'priority', 'created_at']
    list_filter = ['insight_type', 'priority', 'created_at']
    search_fields = ['title', 'report__title']
    readonly_fields = ['created_at']

    fieldsets = (
        ('Basic Information', {
            'fields': ('report', 'insight_type', 'title', 'description', 'priority')
        }),
        ('Insight Data', {
            'fields': ('insight_data',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

@admin.register(ReportTemplate)
class ReportTemplateAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'model_task_type', 'is_active', 'created_at']
    list_filter = ['model_task_type', 'is_active', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']

    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'model_task_type', 'is_active')
        }),
        ('Template Configuration', {
            'fields': ('template_config', 'chart_types', 'insight_types'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
