package com.llm_modifier.fearless_gpus;

import com.llm_modifier.fearless_gpus.security.JwtAuthenticationFilter;
import com.llm_modifier.fearless_gpus.security.JwtService;
import io.jsonwebtoken.JwtException;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.core.userdetails.UserDetailsService;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.*;

class JwtAuthenticationFilterTest {

    @Test
    void expiredOrInvalidTokenReturnsUnauthorized() throws Exception {
        JwtService jwtService = mock(JwtService.class);
        when(jwtService.extractUsername("expired-token")).thenThrow(new JwtException("expired"));
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter(
                jwtService, mock(UserDetailsService.class)
        );
        MockHttpServletRequest request = new MockHttpServletRequest("DELETE", "/api/datasets/1");
        request.addHeader("Authorization", "Bearer expired-token");
        MockHttpServletResponse response = new MockHttpServletResponse();

        filter.doFilter(request, response, mock(jakarta.servlet.FilterChain.class));

        assertEquals(401, response.getStatus());
        assertTrue(response.getContentAsString().contains("Session expired"));
    }
}
